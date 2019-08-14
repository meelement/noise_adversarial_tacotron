from tqdm import tqdm
from pathlib import Path
from os.path import expanduser
import librosa
from os import makedirs
import librosa.filters
import numpy as np
import scipy


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(path, sample_rate, wav.astype(np.int16))


def cut_wav(wave, silence_unit=16000 // 5, threshold=0.05):
    """
    A naive left and right noise cut for VCTK dataset, this dataset has very long silence on start and end.
    Forward attention does not work well with long silences at beginning of clips.
    :param wave: FloatTensor [Samples]
    :param silence_unit: signal bin length
    :param threshold: threshold for cutting off silence.
    :return: processed wave.
    """
    x = wave
    wave = wave[:len(wave) // silence_unit * silence_unit]
    wave = wave.reshape(-1, silence_unit)
    energy = wave ** 2
    energy = energy.sum(axis=-1)
    L = energy.min()
    R = energy.max()
    B = (R - L) * threshold + L
    cut = 0
    for cut in range(0, len(energy)):
        if energy[cut] >= B: break
    cut = cut * silence_unit
    x = x[cut:]
    return x


def process_vctk(
    source=Path(expanduser("~/datasets/vctk/wav16/")),
    target=Path(expanduser("~/datasets/vctk/cut16/")),
    sr=16000
):
    makedirs(str(target), exist_ok=True)
    for path in tqdm(source.glob("*.wav")):
        try:
            wave = load_wav(path, sr)
            wave = cut_wav(wave)
            if len(wave) < sr: continue
            save_wav(wave, str(target / path.name), sr)
        except Exception as e:
            print("VCTK Error ", str(e))


if __name__ == '__main__':
    print("Beginning Cutting Silence from VCTK Dataset")
    process_vctk()
    print("Process Finished")
