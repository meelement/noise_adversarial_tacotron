from pathlib import Path
import torch
import hp
from threading import Thread
from queue import Queue
import numpy as np
from audio.audio_io import load_to_torch
from audio.sfx import mix_with_snr
from torch.nn.utils.rnn import pad_sequence
from utils.text import text_to_sequence
from random import randint, sample, random
from .vctk_meta import vctk_meta


class CHIMELoader:
    """
    Noise is quite small (500 hours of 16KHz Float)
    """
    def __init__(self, wave_path, buffered=True):
        self.wave_path = wave_path = list(wave_path.glob("*.wav"))
        self.buffered = buffered
        self.buffer = {}
        print("CHiME : ", len(self.wave_path))

    def __len__(self):
        return len(self.wave_path)

    def __cache__(self, item):
        if item in self.buffer:
            return self.buffer[item]
        else:
            return self.__read__(item)

    def __read__(self, item):
        return load_to_torch(str(self.wave_path[item]), hp.sampling_rate)

    def __getitem__(self, item: int):
        return self.__cache__(item)

    def sample(self):
        id = randint(0, len(self) - 1)
        return self[id]


class VCTKLoader:
    def __init__(self, wave_path, txt_path):
        wave_path = wave_path.glob("*.wav")
        txt_path = txt_path.glob("*.txt")
        id_wave_path = self.id_wave_path = {}
        id_txt_path = self.id_txt_path = {}
        for file in wave_path:
            id_wave_path[file.stem] = file

        for file in txt_path:
            id_txt_path[file.stem] = file

        ids = self.ids = []

        for id in id_wave_path.keys():
            if id in id_txt_path:
                ids.append(id)

        print("Found VCTK Wave ", len(self))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            txt_path = self.id_txt_path[id]
            wave_path = self.id_wave_path[id]
            with open(str(txt_path), 'r') as f:
                text = f.read().strip()
            speaker = int(id[1:4])
            male = vctk_meta[speaker][1] == "M"
            speaker = speaker - 223
            wave = load_to_torch(str(wave_path), hp.sampling_rate)
            return text.strip(), wave, speaker, male

        elif isinstance(item, slice):
            items = []
            for idx in range(item.start, item.stop):
                items.append(self[idx])
            return items

    def sample(self):
        while True:
            id = randint(0, len(self) - 1)
            text, wave, speaker, male = self[id]
            if len(wave) > hp.min_sample_length and len(wave) < hp.max_sample_length:
                return text, wave, speaker, male


class LJSpeechLoader:
    def __init__(self, wave_path, txt_path):
        wave_path = list(wave_path.glob("*.wav"))
        self.id_text = id_text = {}
        self.ids = ids = []
        self.id_wave_path = id_wave_path = {}
        with open(str(txt_path), 'r') as f:
            for line in f:
                l = line.strip().split('|')
                id_text[l[0]] = l[2]

        for file in wave_path:
            id_wave_path[file.stem] = str(file)
            if file.stem in id_text:
                ids.append(file.stem)

        print("Found LJSpeech Wave ", len(self))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item:int):
        """
        Speaker ID is defined to be one here.
        :param item: int of index
        :return: "hello, world.", WAVE Torch CPU Float, speaker_id in int, False as female
        """
        id = self.ids[item]
        text = self.id_text[id]
        wave_path = self.id_wave_path[id]
        wave = load_to_torch(wave_path, hp.sampling_rate)
        return text.strip(), wave, 1, False

    def sample(self):
        while True:
            id = randint(0, len(self) - 1)
            text, wave, speaker, male = self[id]
            if len(wave) > hp.min_sample_length and len(wave) < hp.max_sample_length:
                return text, wave, speaker, male


class MixingLoader:
    def __init__(self, loaders, ratio):
        """
        Two input must be of the same length
        Simple implementation that does not allow change of sampling ratio.
        :param loaders: list of loaders like above two
        :param ratio: list of int [1, 2] for ratios
        """
        self.loaders = loaders
        self.ratio = ratio
        self.total_ratio = sum(ratio)
        self.loader_lottery_pool = []
        for loader, cnt in zip(loaders, ratio):
            for _ in range(cnt):
                self.loader_lottery_pool.append(loader)
        self.lengths = [len(loader) for loader in loaders]
        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    def sample(self):
        loader = sample(self.loader_lottery_pool, 1)
        return loader[0].sample()


class NoiseAugmentLoader:
    def __init__(self, speech_loader, noise_loader):
        self.speech_loader = speech_loader
        self.noise_loader = noise_loader

    def __len__(self):
        return len(self.speech_loader)

    def sample(self, noise_augment_probability=hp.noise_augment_probability):
        augment = random() < noise_augment_probability
        text, wave, speaker, male = self.speech_loader.sample()
        if augment:
            noise_wave = self.noise_loader.sample()
            begin = randint(0, len(noise_wave) - len(wave) - 1)
            noise_segment = noise_wave[begin: len(wave) + begin]
            mixed_wave = mix_with_snr(wave, noise_segment, randint(5, 25))
            return text, mixed_wave, speaker, male, True
        else:
            return text, wave, speaker, male, False


class BinnedBatchLoader:
    def __init__(
            self,
            q_size,
            n_loading_threads,
            stft: torch.nn.Module,
            redundancy=5,
            device=hp.device
    ):
        self.loader = NoiseAugmentLoader(
            speech_loader=MixingLoader(
                [
                    LJSpeechLoader(hp.ljs_wav_path, hp.ljs_text_path),
                    VCTKLoader(hp.vctk_wav_path, hp.vctk_text_path)
                ], [1, 1] if hp.debug else[10, 90]
            ),
            noise_loader=CHIMELoader(hp.part_chime_path)
        )

        self.stft = stft
        # Loading From File System
        self.device = device
        self.loading_threads = []
        self.loading_queue = Queue(maxsize=q_size)
        for _ in range(n_loading_threads):
            self.loading_threads.append(Thread(target=self.loading_thread))
            self.loading_threads[-1].start()

        self.r = 1
        self.batch_size = 1
        self.redundancy = redundancy
        self.device = device
        self.blocking_threads = Thread(target=self.blocking_thread)
        self.blocking_queue = Queue(maxsize=5)
        self.blocking_threads.start()

    def get_state(self):
        return self.loading_queue.qsize(), self.loading_queue.maxsize, self.blocking_queue.qsize(), self.blocking_queue.maxsize

    def set_state(self, batch_size, r):
        self.batch_size = batch_size
        self.r = r

    def get_batch(self):
        return self.blocking_queue.get()

    def loading_thread(self):
        while True:
            try:
                text, wave, speaker, male, augmented = self.loader.sample()
                phoneme = text_to_sequence(text, hp.cleaner_names)
                phoneme = torch.from_numpy(np.int64(phoneme))
                self.loading_queue.put((phoneme.to(self.device, non_blocking=True), wave.to(self.device, non_blocking=True), speaker, augmented))
            except Exception as e:
                print("Loading Thread Error", str(e))

    def get_n(self, n:int):
        """
        Getting list of phonemes and waves from loaded instances
        :param n: int
        :return: list of tuple(phone, wave, wave_len)
        wave_len is for sorting.
        """
        items = []
        for _ in range(n):
            phoneme, wave, speaker, augmented = self.loading_queue.get(block=True)
            items.append((phoneme, wave, speaker, augmented, len(wave)))
        return items

    def blocking_thread(self):
        while True:
            try:
                items = self.get_n(self.batch_size * self.redundancy)
                # sort items based on the length
                # This has to be reversed since we are using the function pack_padded_sequence.
                items = sorted(items, key=lambda x: x[-1], reverse=True)
                batch_size = self.batch_size
                r = self.r
                for cnt in range(self.redundancy):
                    if self.batch_size != batch_size or self.r != r: break
                    scatter = items[cnt * batch_size: (cnt + 1) * batch_size]
                    phone, wave, speaker, augmented, wavelen = zip(*scatter)
                    block = self.packing_batch(phone, wave, speaker, augmented, r)
                    self.blocking_queue.put(block)
            except Exception as e:
                print("Blocking Thread Error", str(e))

    def packing_batch(self, phone, wave, speaker, augmented, r):
        """
        :param phone: list of english phone.
        :param wave: list of GPU torch FloatTensor
        :param speaker: list of int as speaker ids
        :param augmented: list of bools whether the speech is augmented
        :return: Phone, Wave, Speaker, Augmented, PhoneLength, WaveLength, FrameLength
        All R normalized and zero padded.
        """
        phone_lengths = [len(t) for t in phone]
        wave_lengths = [len(t) for t in wave]
        frame_lengths = [self.stft.sample_to_frame(t) for t in wave_lengths]
        norm_frame_lengths = [t // r * r for t in frame_lengths]
        norm_sample_lengths = [self.stft.frame_to_sample(t) for t in norm_frame_lengths]
        norm_wave = [a[:l] for a, l in zip(wave, norm_sample_lengths)]
        Wave = pad_sequence(norm_wave, batch_first=True)
        Phone = pad_sequence(phone, batch_first=True)
        PhoneLength = torch.LongTensor(phone_lengths).to(self.device)
        WaveLength = torch.LongTensor(norm_sample_lengths).to(self.device)
        FrameLength = torch.LongTensor(norm_frame_lengths).to(self.device)
        Speaker = torch.LongTensor(speaker).to(self.device)
        Augmented = torch.LongTensor(augmented).to(self.device)
        return Phone, Wave, Speaker, Augmented, PhoneLength, WaveLength, FrameLength, r

    def __len__(self):
        return len(self.loader)
