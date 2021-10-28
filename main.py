import argparse
import warnings
import os
import random
import numpy as np
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SpeechDataset
from model.DCCRN import set_model
from utils.losses import *
from utils.metrics import pesq_score
from utils.utils import generate_wav

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=40, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--lr-decay', default=0.1)
parser.add_argument('--weight-decay', default=1e-5)

parser.add_argument('--arch', type=str, default="CL")

parser.add_argument('--clean-train-dir', type=str, default="dataset/train/clean_trainset")
parser.add_argument('--noisy-train-dir', type=str, default="dataset/train/noisy_trainset")
parser.add_argument('--clean-valid-dir', type=str, default="dataset/valid/clean_validset")
parser.add_argument('--noisy-valid-dir', type=str, default="dataset/valid/noisy_validset")
parser.add_argument('--clean-test-dir', type=str, default="dataset/test/clean_testset")
parser.add_argument('--noisy-test-dir', type=str, default="dataset/test/noisy_testset")

parser.add_argument('--sample-rate', type=int, default=48000, help="STFT hyperparam")
parser.add_argument('--max-len', type=int, default=165000)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')
# generate
parser.add_argument('--generate', '-g', default=False, action='store_true')
parser.add_argument('--denoising-file', type=str, help="denoising 하고 싶은 파일경로")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

summary = SummaryWriter()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # STFT 인자
    sampling_rate = args.sample_rate
    N_FFT = sampling_rate * 64 // 1000 + 4
    # N_FFT = int(.02 * args.sample_rate)

    HOP_LENGTH = sampling_rate * 16 // 1000 + 4
    # HOP_LENGTH = int(.01 * args.sample_rate)
    # print(HOP_LENGTH)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Model
    model = set_model(mode=args.arch, args=args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("this")
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Optimizer / criterion(wSDR)
    criterion = SISNRLoss().cuda(args.gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5),
                    int(args.epochs * 0.7),
                    int(args.epochs * 0.9)],
        gamma=args.lr_decay
    )

    best_PESQ = -1e10

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            best_PESQ = checkpoint['PESQ']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        # 만약 Dataparallel 으로 저장했을 시 이 코드 사용
        # stat_dict = torch.load(args.resume, map_location="cuda:0")
        #
        # new_state_dict = OrderedDict()
        # for k, v in stat_dict.items():
        #     name = k[7:] # remove 'module'
        #     new_state_dict[name] = v
        #
        # model.load_state_dict(new_state_dict)

    # generate wav file
    if args.generate:
        generate_wav(model, args.max_len, args)
        print("Generate Denoising File")
        return

    # Dataset path
    mixed_train_dir = Path(args.noisy_train_dir)
    clean_train_dir = Path(args.clean_train_dir)

    mixed_valid_dir = Path(args.noisy_valid_dir)
    clean_valid_dir = Path(args.clean_valid_dir)

    mixed_test_dir = Path(args.noisy_test_dir)
    clean_test_dir = Path(args.clean_test_dir)

    # 파일 리스트
    mixed_train_files = sorted(list(mixed_train_dir.rglob('*.wav')))
    clean_train_files = sorted(list(clean_train_dir.rglob('*.wav')))

    mixed_valid_files = sorted(list(mixed_valid_dir.rglob('*.wav')))
    clean_valid_files = sorted(list(clean_valid_dir.rglob('*.wav')))

    mixed_test_files = sorted(list(mixed_test_dir.rglob('*.wav')))
    clean_test_files = sorted(list(clean_test_dir.rglob('*.wav')))

    # Dataset
    train_dataset = SpeechDataset(args, mixed_train_files, clean_train_files, args.max_len)
    # valid_dataset = SpeechDataset(args, mixed_valid_files, clean_valid_files, args.max_len)
    test_dataset = SpeechDataset(args, mixed_test_files, clean_test_files, args.max_len)

    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        print("Sampler")
    else:
        train_sampler = None

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
    #                                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    # Evaluate
    if args.evaluate:
        PESQ = validate(test_loader, model, criterion, N_FFT, HOP_LENGTH, args, summary)
        print(f" | PESQ: {PESQ:.4f}".format(
            PESQ=PESQ
        ))
        return

    # Train

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, scheduler, epoch, N_FFT, HOP_LENGTH, args, summary)

        print("--validate--")
        PESQ = validate(test_loader, model, criterion, N_FFT, HOP_LENGTH, args, summary, epoch)

        print(f" | PESQ: {PESQ:.4f}".format(
            PESQ=PESQ
        ))

        if best_PESQ < PESQ:  # 현재 PESQ 더 클시
            print("Found better validated model")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                torch.save({
                    'epoch': epoch + 1,
                    'PESQ': best_PESQ,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, "saved_models/checkpoint_%d.pth" % (epoch + 1))
            best_PESQ = PESQ


def train(train_loader, model, criterion, optimizer, scheduler, epoch, n_fft, hop_length, args, summary):
    model.train()

    end = time.time()
    # Dataset return x_noisy_stft, x_clean_stft
    for i, (mixed, target) in enumerate(train_loader):
        mixed = mixed.cuda(args.gpu, non_blocking=True) # [batch=2, channel=1, time * SR=165000]
        target = target.cuda(args.gpu, non_blocking=True)#
        # print("target: ", target.size())

        spec, wav = model(mixed)
        # print("spec: ", spec.size()) # istft_spec[batch=2, dim=512, length=1653]
        # print("wav: ", wav.size()) # wav[batch=2, max_len=165000]
        loss = criterion(wav, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 ==0:
            niter = epoch * len(train_loader) + i
            summary.add_scalar('Train/Loss', loss.item(), niter)
            summary.add_scalar('Train/lr', get_lr(optimizer), niter)

        if i % args.print_freq == 0:
            print(get_lr(optimizer))
            print(" Epoch [%d][%d/%d] | loss: %f" % (epoch + 1, i, len(train_loader), loss))

    scheduler.step()
    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(dataloader, model, criterion, n_fft, hop_length, args, summary, epoch=0):
    model.eval()
    # loss와 score를 동시에 구하는 함수로 대체하였음
    score = pesq_score(model, dataloader, criterion, args, n_fft, hop_length, summary, epoch)

    return score

if __name__ == "__main__":
    main()














