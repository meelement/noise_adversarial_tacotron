import hp
from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa
import torch
import librosa.filters
import numpy as np
import scipy
from random import randint
from os import makedirs


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(path, sample_rate, wav.astype(np.int16))


def get_segments(source, length, count):
    begins = []
    l = len(source)
    for _ in range(count):
        begins.append(randint(0, l - length - 1))
    segments = []
    for begin in begins:
        segments.append(source[begin: begin + length])
    return segments


def process_chime(
    source=hp.whole_chime_path,
    target=hp.part_chime_path,
    sr=16000,
    duration=30,
    count=10
):
    """
    Randomly picking segments from CHiME dataset, since full dataset is not necessary in our case.
    :param source:
    :param target:
    :param sr:
    :param duration:
    :param count:
    :return:
    """
    makedirs(str(target), exist_ok=True)
    for path in tqdm(source.glob("*.wav")):
        wave = load_wav(path, sr)
        if len(wave) < sr * 30: continue
        waves = get_segments(wave, duration * sr, count)
        for i, wave in enumerate(waves, 1):
            save_wav(wave, str(target / f"{path.stem}_{i}.wav"), sr)


if __name__ == '__main__':
    print("Beginning segmenting CHiME4 noises.")
    process_chime()
    print("Processing Finished")
