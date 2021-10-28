import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    # s1: estimated time domain waveform( mask 입혀진 waveform )
    # s2: clean time domain waveform( Ground Truth 느낌 )
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target

    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)

    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


class SISNRLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(SISNRLoss, self).__init__()
        self.eps = eps

    def forward(self, s1, s2):
        # s1: estimated time domain waveform( mask 입혀진 waveform )
        # s2: clean time domain waveform( Ground Truth 느낌 )
        return -(si_snr(s1, s2, eps=self.eps))


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        batch, dim, time = x.shape
        # todo 0 의미
        y[:, 0, :] = 0
        y[:, dim // 2, :] = 0

        return F.mse_loss(x, y, reduction='mean') * dim


class MAELoss(nn.Module):

    def __init__(self, stft):
        super(MAELoss, self).__init__()
        self.stft = stft

    def forward(self, x, y):
        target_spec, target_phase = self.stft(y)
        batch, dim, time = x.shape

        return torch.mean(torch.abs(x - target_spec)) * dim

if __name__ == "__main__":
    a = SISNRLoss().cuda()
    b = torch.randn(2,165000)
    c = torch.randn(2,165000)
    print(a(b,c))