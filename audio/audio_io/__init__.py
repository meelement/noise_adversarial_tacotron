import librosa
import torch
import librosa.filters
import numpy as np
import scipy
import struct
from pyaudio import PyAudio, paInt16


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def load_to_torch(path, sample_rate):
    wave = load_wav(path, sample_rate)
    return torch.from_numpy(wave).float()


def save_wav(wav, path, sample_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(path, sample_rate, wav.astype(np.int16))


def save_from_torch(wav, path, sample_rate):
    wav = wav.detach().cpu().numpy()
    save_wav(wav, path, sample_rate)


def show_notebook(wave:np.ndarray, sr=16000):
    from IPython.display import Audio
    wave = np.int16(wave * 65536)
    return Audio(wave, rate=sr)
