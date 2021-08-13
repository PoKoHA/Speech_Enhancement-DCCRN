import numpy as np
import librosa
import matplotlib.pyplot as plt

import torch
import torchaudio
import torchaudio.functional as F

from torch.utils.data import Dataset

class SpeechDataset(Dataset):

    def __init__(self, args, noisy_files, clean_files, max_len):
        super(SpeechDataset, self).__init__()
        self.args = args

        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # fixed len
        self.max_len = max_len
        self.datasize = len(self.noisy_files)

    def __len__(self):
        return self.datasize

    def load_sample(self, file):
        waveform, sr = torchaudio.load(file)

        return waveform

    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)

        return output

    def __getitem__(self, idx):
        x_clean = self.load_sample(self.clean_files[idx])
        # x_clean = self.load_sample("dataset/56spk/noisy_trainset_56spk_wav/p308_420.wav")
        x_noisy = self.load_sample(self.noisy_files[idx])
        # x_noisy = self.load_sample("dataset/56spk/noisy_trainset_56spk_wav/p308_420.wav")

        # print(self.clean_files[idx], x_clean.size())
        # print(self.noisy_files[idx], x_noisy.size())
        # plt.figure(figsize=(15, 5))
        # plt.plot(x_clean.squeeze(0).cpu().numpy())
        # plt.title(self.clean_files[idx])
        # plt.show()
        # print(x_noisy.size())

        # padding / cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        # plt.figure(figsize=(15, 5))
        # plt.plot(x_clean.squeeze(0).cpu().numpy())
        # plt.title(self.clean_files[idx])
        # plt.show()
        # print(x_noisy.size())

        return x_noisy, x_clean
