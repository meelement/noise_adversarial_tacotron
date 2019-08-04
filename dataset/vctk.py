from pathlib import Path
import torch
import hp
from threading import Thread
from queue import Queue
import numpy as np
from audio.audio_io import load_to_torch
from time import sleep
from torch.nn.utils.rnn import pad_sequence
from utils.text import text_to_sequence
from random import randint


class MetaLoader:
    def __init__(self, wave_path: Path, txt_path: Path):
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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item: int):
        return self.id_txt_path[self.ids[item]], self.id_wave_path[self.ids[item]]


class TextWaveLoader(MetaLoader):
    def __init__(self, wave_path, txt_path):
        super(TextWaveLoader, self).__init__(wave_path, txt_path)

    def __getitem__(self, item:int):
        """
        :param item: int of index
        :return: "hello, world.", WAVE Torch CPU Float
        """
        txt_path, wave_path = super(TextWaveLoader, self).__getitem__(item)
        text = wave = None
        with open(str(txt_path), 'r') as f:
            text = f.read().strip()
        wave = load_to_torch(wave_path, hp.sampling_rate)
        return text.strip(), wave


class BinnedBatchLoader(TextWaveLoader):
    # TODO : Ensure that this loader sends everything to GPU async.
    def __init__(
            self,
            wave_path,
            txt_path,
            q_size,
            n_loading_threads,
            stft: torch.nn.Module,
            redundancy=5,
            device="cuda:0"
    ):
        super(BinnedBatchLoader, self).__init__(wave_path, txt_path)
        self.stft = stft
        # Loading From File System
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
        for _ in range(self.blocking_queue.maxsize):
            self.blocking_queue.get()

    def get_batch(self):
        return self.blocking_queue.get()

    def loading_thread(self):
        while True:
            try:
                id = randint(0, len(self) - 1)
                text, wave = self[id]
                phoneme = text_to_sequence(text, hp.cleaner_names)
                phoneme = torch.from_numpy(np.int64(phoneme))
                self.loading_queue.put((phoneme.to(self.device, non_blocking=True), wave.to(self.device, non_blocking=True)))
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
            phoneme, wave = self.loading_queue.get(block=True)
            items.append((phoneme, wave, len(wave)))
        return items

    def blocking_thread(self):
        while True:
            try:
                items = self.get_n(self.batch_size * self.redundancy)
                # sort items based on the length
                items = sorted(items, key=lambda x: x[2])
                batch_size = self.batch_size
                r = self.r
                for cnt in range(self.redundancy):
                    if self.batch_size != batch_size or self.r != r:
                        while not self.blocking_queue.empty():
                            self.blocking_queue.get()
                        break
                        print("Cleaned Queue ... ")

                    scatter = items[cnt * batch_size: (cnt + 1) * batch_size]
                    phone, wave, _ = zip(*scatter)
                    block = self.packing_batch(phone, wave)
                    self.blocking_queue.put(block)
            except Exception as e:
                print("Blocking Thread Error", str(e))

    def packing_batch(self, phone, wave):
        """
        :param phone: list of english phone.
        :param wave: list of GPU torch FloatTensor
        :return: Phone, Wave, PhoneLength, WaveLength, FrameLength
        All R normalized and zero padded.
        """
        phone_lengths = [len(t) for t in phone]
        wave_lengths = [len(t) for t in wave]
        frame_lengths = [self.stft.sample_to_frame(t) for t in wave_lengths]
        norm_frame_lengths = [t // self.r * self.r for t in frame_lengths]
        norm_sample_lengths = [self.stft.frame_to_sample(t) for t in norm_frame_lengths]
        norm_wave = [a[:l] for a, l in zip(wave, norm_sample_lengths)]
        Wave = pad_sequence(norm_wave, batch_first=True)
        Phone = pad_sequence(phone, batch_first=True)
        PhoneLength = torch.LongTensor(phone_lengths)
        WaveLength = torch.LongTensor(norm_sample_lengths)
        FrameLength = torch.LongTensor(norm_frame_lengths)
        return Phone, Wave, PhoneLength, WaveLength, FrameLength
