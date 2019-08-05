import torch
import hp
from utils.text.symbols import symbols
from tacotron import Tacotron
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
from audio.spec.mel import MelSTFT
from audio.audio_io import save_from_torch
from os import makedirs
from os.path import exists


class Synthesizer:
    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = checkpoint_path
        assert exists(checkpoint_path)
        self.device = torch.device(device)

        print('\nInitialising Tacotron Model...\n')

        # Instantiate Tacotron Model
        self.tts_model = tts_model = Tacotron(
            embed_dims=hp.embed_dims,
            num_chars=len(symbols),
            encoder_dims=hp.encoder_dims,
            decoder_dims=hp.decoder_dims,
            n_mels=hp.n_mels,
            fft_bins=hp.fft_bins,
            postnet_dims=hp.postnet_dims,
            encoder_K=hp.encoder_K,
            lstm_dims=hp.lstm_dims,
            postnet_K=hp.postnet_K,
            num_highways=hp.num_highways,
            dropout=hp.dropout
        ).to(device=self.device)

        print("\nInitializing STFT Model...\n")

        self.stft = stft = MelSTFT(
            filter_length=hp.n_fft,
            hop_length=hp.hop_length,
            win_length=hp.win_length,
            n_mel_channels=hp.n_mels,
            sampling_rate=hp.sampling_rate,
            mel_fmin=hp.min_f,
            mel_fmax=hp.max_f
        ).to(device=self.device)

        tts_model.restore(self.checkpoint_path)
        tts_model.eval()
        # print some information
        self.tts_k = tts_model.get_step() // 1000

        r = tts_model.get_r()

        simple_table([
            (f'Tacotron(r={r})', str(self.tts_k) + 'k'),
            ("Sample Rate", hp.sampling_rate),
            ("NFFT", hp.n_fft),
            ("NMel", hp.n_mels)
        ])

    def synthesis(self, text, id="test", wave_path="log/synthesis/wave/", plot_path="log/synthesis/plot/"):
        makedirs(wave_path, exist_ok=True)
        makedirs(plot_path, exist_ok=True)
        phone = text_to_sequence(text.strip(), hp.cleaner_names)
        with torch.no_grad():
            mel, linear, attention = self.tts_model.generate(phone)

        audio_path = f'{wave_path}_GL_input_{text[:10]}_{self.tts_k}k.wav'
        atten_path = f"{plot_path}_Attention_input_{text[:10]}_{self.tts_k}k"

        save_attention(attention, atten_path)

        print(list(linear.shape))
        wave = self.stft.inverse_linear(linear, iteration=40)[0]
        save_from_torch(wave, audio_path, hp.sampling_rate)
