"""
PreTrained된 model을 이용하여 Denoising 된 파일들을 Generate
"""
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import librosa

import torch
import torchaudio
from model.conv_stft import *


def display_spectrogram(x, title):
    plt.figure(figsize=(15, 10))
    plt.pcolormesh(x[0], cmap='hot')
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.show()


def generate_wav(model, max_len, args):
    model.eval()
    file_list = os.listdir(args.denoising_file)
    stft = ConvSTFT(400, 100, 512, 'hanning', 'complex', fix=True).cuda()
    file_list = sorted(file_list)

    for idx in range(len(file_list)):
        name = file_list[idx]
        mixed = os.path.join(args.denoising_file, name)
        waveform, _ = torchaudio.load(mixed)
        # plt.figure(figsize=(15, 5))
        # plt.plot(waveform.squeeze(0).cpu().numpy())
        # plt.title("Origin")
        # plt.show()
        # print(waveform.size())
        waveform = waveform.numpy()

        current_len = waveform.shape[1]
        pad = np.zeros((1, max_len), dtype='float32')
        pad[0, -current_len:] = waveform[0, :max_len]

        input = torch.from_numpy(pad).cuda(args.gpu)

        input_stft = stft(input)
        r = input_stft[:, :257]
        i = input_stft[:, 257:]
        input_stft = torch.stack([r, i], dim=-1)

        input_r = input_stft[..., 0]
        input_i = input_stft[..., 1]
        input_mag = torch.sqrt(input_r ** 2 + input_i ** 2)
        input_phase = torch.atan2(input_i, input_r)
        db_1 = librosa.amplitude_to_db(input_r.cpu().detach().numpy())
        db_2 = librosa.amplitude_to_db(input_i.cpu().detach().numpy())
        db_3 = librosa.amplitude_to_db(input_mag.cpu().detach().numpy())
        db_4 = librosa.amplitude_to_db(input_phase.cpu().detach().numpy())
        # display_spectrogram(db_1, name+"/input real")
        # display_spectrogram(db_2, name+"/input imag")
        display_spectrogram(db_3, name + "/input mag")
        display_spectrogram(db_4, name + "/input phase")

        with torch.no_grad():
            spec, wav = model(input)

            a = stft(wav)
            r = a[:, :257]
            i = a[:, 257:]
            a = torch.stack([r, i], dim=-1)

        input_r = a[..., 0]
        input_i = a[..., 1]
        input_mag = torch.sqrt(input_r ** 2 + input_i ** 2)
        input_phase = torch.atan2(input_i, input_r)
        db_1 = librosa.amplitude_to_db(input_r.cpu().detach().numpy())
        db_2 = librosa.amplitude_to_db(input_i.cpu().detach().numpy())
        db_3 = librosa.amplitude_to_db(input_mag.cpu().detach().numpy())
        db_4 = librosa.amplitude_to_db(input_phase.cpu().detach().numpy())
        # display_spectrogram(db_1, name + "/denoising real")
        # display_spectrogram(db_2, name + "/denoising imag")
        display_spectrogram(db_3, name + "/denoising mag")
        display_spectrogram(db_4, name + "/denoising phase")
        # plt.figure(figsize=(15,5))
        # plt.plot(pred.squeeze(0)[-current_len:].cpu().numpy())
        # plt.title("Denoising")
        # plt.show()
        # print("A", pred.size())
        output = os.path.join("output", name)
        sf.write(output, wav.squeeze(0)[0][-current_len:].cpu().numpy(), samplerate=48000, format='WAV', subtype='PCM_16')


##############################################
def display_feature(input, title):
    x = input[0]
    size = x.size(0) # Channel
    plt.figure(figsize=(50, 10))
    plt.title(title)
    x = x.cpu().detach().numpy()
    x = x[:, ::-1, :]
    a = 0
    for i in range(size):
        a += x[i]
    # plt.subplot(16, 32, i+1)
    plt.imshow(a, cmap='gray')
    plt.axis("off")

    plt.show()
    plt.close()