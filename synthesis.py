import torch
import hp
from utils.text.symbols import symbols
from model.tacotron import Tacotron
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
from audio.spec.mel import MelSTFT
from audio.audio_io import save_from_torch
from os import makedirs
from os.path import exists


class Synthesizer:
    def __init__(self, checkpoint_path, device="cuda"):
        self.checkpoint_path = checkpoint_path
        assert exists(checkpoint_path)
        self.device = torch.device(device)

        print('\nInitialising Tacotron Model...\n')

        # Instantiate Tacotron Model
        self.tacotron = tts_model = Tacotron(
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
            dropout=hp.dropout,
            speaker_latent_dims=hp.speaker_latent_dims,
            speaker_encoder_dims=hp.speaker_encoder_dims,
            n_speakers=hp.n_speakers,
            noise_latent_dims=hp.noise_latent_dims,
            noise_encoder_dims=hp.noise_encoder_dims
        ).to(device=self.device)

        print("\nInitializing STFT Model...\n")

        self.stft = MelSTFT(
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
            ("NMel", hp.n_mels),
            ("Speakers", hp.n_speakers),
            ("SPKD", hp.speaker_latent_dims),
            ("NOID", hp.noise_latent_dims),
        ])

    def inference_speaker_noise(self, wave):
        """
        :param wave: Any [Time] in [-1, 1]
        :return: [Speaker], [Noise]
        """
        with torch.no_grad():
            device = self.device
            wave = torch.as_tensor(wave, dtype=torch.float, device=device).unsqueeze(0) # [Batch, Time]
            mel, linear = self.stft.transform(wave) # [1, NMel / Linear, Frame]
            l = torch.as_tensor([mel.size(-1)], dtype=torch.long, device=device)
            speaker_latent = self.tacotron.batch_inference_speaker(mel, l) # [Batch, Speaker]
            noise_latent = self.tacotron.batch_inference_noise(mel, l) # [Batch, Noise]
            return speaker_latent.squeeze(0), noise_latent.squeeze(0)

    def batch_inference_speaker_noise(self, waves, wave_lens=None):
        """
        :param waves: Any [Batch, Time] in [-1, 1]
        :param wave_lens: AnyLong [Batch] used to inference the length of the mel
        :return: Speaker FloatTensor[Batch, Speaker], Noise FloatTensor[Batch, Noise]
        """
        with torch.no_grad():
            device = self.device
            waves = torch.as_tensor(waves, dtype=torch.float, device=device) # [Batch, Time]
            wave_lens = torch.as_tensor(wave_lens, dtype=torch.long, device=device) # [Batch]
            mel, linear = self.stft.transform(waves) # [Batch, NMel / Linear, Frame]
            l = self.stft.sample_to_frame(wave_lens) # Long[Batch]
            speaker_latent = self.tacotron.batch_inference_speaker(mel, l) # [Batch, Speaker]
            noise_latent = self.tacotron.batch_inference_noise(mel, l) # [Batch, Noise]
            return speaker_latent, noise_latent

    def synthesis(self, text, speaker_embedding, noise_embedding, wave_path="log/synthesis/wave/", plot_path="log/synthesis/plot/"):
        """
        TODO: Provide Batch Synthesis
        :param text: "hello, world"
        :param speaker_embedding: Any[Speaker]
        :param noise_embedding: Any [Noise]
        :param wave_path: "log/synthesis/wave"
        :param plot_path: "log/synthesis/plot"
        :return: FloatTensor [Time] for wave, FloatTensor [Encoder, Decoder] for attention
        """
        with torch.no_grad():
            makedirs(str(wave_path), exist_ok=True)
            makedirs(str(plot_path), exist_ok=True)
            phone = text_to_sequence(text.strip(), hp.cleaner_names)
            mel, linear, attention = self.tacotron.generate(phone, speaker_embedding, noise_embedding)

            audio_path = f'{wave_path}_GL_input_{text[:10]}_{self.tts_k}k.wav'
            atten_path = f"{plot_path}_Attention_input_{text[:10]}_{self.tts_k}k"

            save_attention(attention, atten_path)

            print(list(linear.shape))
            wave = self.stft.inverse_linear(linear, iteration=40)[0]
            save_from_torch(wave, audio_path, hp.sampling_rate)

            return wave, attention
