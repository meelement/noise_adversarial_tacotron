import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
import torch

from matplotlib import pyplot as plt


SAMPLE = 44100
TOP = 32767


def load_wave_file_to_numpy(file_path, sample_rate=SAMPLE, *args, **kwargs):
    return librosa.load(file_path, sr=sample_rate, *args, **kwargs)


def plot_complex_spectrogram(D: np.ndarray, title="test audio", y_axis="log", cmap="binary", angle_off=False):
    angle = np.angle(D)
    abs = np.abs(D)
    librosa.display.specshow(abs, y_axis=y_axis, cmap=cmap)
    if not angle_off:
        librosa.display.specshow(angle, y_axis=y_axis, alpha=0.2)
    plt.title(f"Log-frequency power spectrogram of {title}")


def plot_wave_and_spectrogram(numpy_wave: np.ndarray, D=None, title="test audio", y_axis="log", cmap="binary", angle_off=False):
    fig = plt.figure(figsize=(12, 8))
    if D is None:
        D = _stft(numpy_wave)

    plt.subplot(2, 1, 1)
    plot_complex_spectrogram(D, title, y_axis, cmap=cmap, angle_off=angle_off)
    plt.subplot(2, 1, 2)
    plt.plot(numpy_wave, lw=0.6)
    plt.title("Waveform")
    plt.margins(x=0)
    plt.plot()
    plt.show()
    plt.close()


def _stft(wave, n_fft=1024,
          hop_length=512,
          ):
    return librosa.stft(wave, n_fft=n_fft, hop_length=hop_length)


def _istft(spec,
           hop_length=512
           ):
    return librosa.istft(spec, hop_length=hop_length)


def split_complex_spectra_into_magnitude_and_phase(spectra):
    return np.abs(spectra), np.angle(spectra)


def merge_magnitude_and_phase_to_complex_spectra(magnitude, phase):
    Real = np.cos(phase) * magnitude
    Imag = np.sin(phase) * magnitude
    Bind = np.zeros_like(magnitude, dtype=complex)
    Bind.real = Real
    Bind.imag = Imag
    return Bind


def griffin_lim(Amp, Ang):
    y = _istft(merge_magnitude_and_phase_to_complex_spectra(Amp, Ang))
    _, Ang = split_complex_spectra_into_magnitude_and_phase(_stft(y))
    return Ang


def apply_magnitude_to_another(source, target):
    Source = _stft(source)
    Target = _stft(target)
    Angle = np.angle(Target)
    Mag = np.abs(Source)
    Real = np.cos(Angle) * Mag
    Imag = np.sin(Angle) * Mag
    Bind = np.zeros_like(Target, dtype=complex)
    Bind.real = Real
    Bind.imag = Imag

    wave = _istft(Bind)
    return wave


def read_advanced_format_to_np(file, format):
    segment = AudioSegment.from_file(file, format)
    wave = segment.get_array_of_samples()
    wave = np.array(wave) / 65536
    left = wave[::2]
    right = wave[1::2]
    wave = left + right
    wave = wave / 3
    return wave


def from_int16_to_float_numpy(arr):
    return arr / 65536


def mono_pad_or_truncate(original, estimate):
    if len(estimate) > len(original):
        return original, estimate[:len(original)]
    else:
        return original[:len(estimate)], estimate


def overlap_click(original, miliseconds, sr=44100):
    for point in miliseconds:
        assert 0 <= point < len(original) / sr, "frame out of range"


def standardize(wave):
    top = np.max(np.abs(wave))
    return wave / top


def overlap_click(original, click_position, sr=44100, click_freq=2000, click_duration=0.5):
    """
    :param click_position: Notice that position should be given in second
    :return: wave
    """
    cwave = librosa.clicks(np.array(click_position), sr=44100, click_freq=4000, click_duration=0.05) / 2
    original, wave = mono_pad_or_truncate(original, cwave)
    return standardize(original + wave)


def mmsecond_to_sample(mseconds, sr=44100):
    return [msecond * 44.1 for msecond in mseconds]


def mfcc(wave: np.ndarray, n_mfcc=40, sr=44100):
    return librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc)


def plot_wave_and_mfcc(wave, sr=44100, n_mfcc=40):
    M = mfcc(wave, sr=sr, n_mfcc=n_mfcc)
    plot_wave_and_spectrogram(wave, D=M, cmap="hsv", y_axis="linear", angle_off=True)

def mel(wave: np.ndarray, n_mels=80, sr=44100):
    return librosa.feature.melspectrogram(wave, sr=sr, n_mels=n_mels)

def torch_raw_stft(wave, n_fft=2048, hop=512):
    """
    :param wave:
    :param n_fft:
    :param hop:
    :return: [B, C, T, 2]
    """
    if not torch.is_tensor(wave):
        wave = torch.tensor(wave)
    return torch.stft(wave, n_fft=n_fft, hop_length=hop)


def torch_mag_stft(wave, n_fft=2048, hop=512):
    """
    :param wave:
    :param n_fft:
    :param hop:
    :return: [B, C, T]
    """
    Wave = torch_raw_stft(wave, n_fft, hop)
    return Wave.pow(2).sum(dim=-1).sqrt()


def torch_spec_to_numpy_complex(Wave):
    """
    :param Wave:
    :param hop:
    :return: You Better Split ME [..., x]
    """
    Real = Wave[..., 0].cpu().numpy()
    Imag = Wave[..., 1].cpu().numpy()
    Combine = np.complex_(Real) + 1.0j * np.complex_(Imag)
    return Combine


def torch_spec_to_numpy_arg_mag(Wave):
    Wave = torch_spec_to_numpy_complex(Wave)
    Mag, Arg = split_complex_spectra_into_magnitude_and_phase(Wave)
    return Mag, Arg

if __name__ == '__main__':
    wave, sr = load_wave_file_to_numpy("test.wav")
    spec = mel(wave)
    plot_wave_and_spectrogram(wave, D=spec, y_axis="linear")
    print(spec.shape)
