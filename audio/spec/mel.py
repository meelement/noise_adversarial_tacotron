import torch
from .stft import STFT
from librosa.filters import mel as librosa_mel_fn
from .functional import dynamic_range_compression, dynamic_range_decompression
import numpy as np


class MelSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=16 * 8, win_length=1024,
                 n_mel_channels=128, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(MelSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis_inv = torch.from_numpy(np.linalg.pinv(mel_basis)).float()
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('mel_basis_inv', mel_basis_inv)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        output += 11.5
        output /= 3
        return output

    def spectral_de_normalize(self, magnitudes):
        magnitudes *= 3
        magnitudes -= 11.5
        output = dynamic_range_decompression(magnitudes)
        return output

    def transform(self, x):
        """
        Forward Transform to generate Mel Scaled Magnitude Spectrogram
        :param x: input signal [Batch, Samples]
        :return: Mel Spectrogram [Batch, Mel Filters, Frames]
        """
        linear, phases = self.stft_fn.transform(x)
        mel_output = torch.matmul(self.mel_basis, linear)
        mel_output = self.spectral_normalize(mel_output)
        linear = self.spectral_normalize(linear)
        return mel_output, linear

    def inverse_mel(self, y, iteration=40):
        """
        Backward Transform to generate Estimated Audio
        :param spec: [Batch, Mel Filters, Frames]
        :return: Estimated Audio [Batch, Samples]
        """
        y = self.spectral_de_normalize(y)
        magnitudes = torch.matmul(self.mel_basis_inv, y)
        return self.stft_fn.griffin_lim(magnitudes, iteration)

    def inverse_linear(self, y, iteration=40):
        """
        Backward Transform to generate Estimated Audio
        :param spec: [Batch, NFFT, Frames]
        :return: Estimated Audio [Batch, Samples]
        """
        y = self.spectral_de_normalize(y)
        return self.stft_fn.griffin_lim(y, iteration)

    def sample_to_frame(self, n):
        return self.stft_fn.sample_to_frame(n)

    def frame_to_sample(self, f):
        return self.stft_fn.frame_to_sample(f)
