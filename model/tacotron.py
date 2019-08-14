import os
import numpy as np
import torch
import torch.nn as nn
import hp
from model.decoder import Decoder
from model.encoder import Encoder
from model.modules import CBHG
from model.encoder import CNNEncoder, CNNRNNEncoder
from model.utils import grad_reverse
from model.modules import Classifier
import torch.distributions as dist


class Tacotron(nn.Module):
    def __init__(
            self,
            embed_dims,
            num_chars,
            encoder_dims,
            decoder_dims,
            n_mels,
            fft_bins,
            postnet_dims,
            encoder_K,
            lstm_dims,
            postnet_K,
            num_highways,
            dropout,
            speaker_latent_dims,
            speaker_encoder_dims,
            n_speakers,
            noise_latent_dims,
            noise_encoder_dims
    ):
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims

        # Standard Tacotron #############################################################
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_K, num_highways, dropout)
        self.encoder_proj = nn.Linear(decoder_dims, decoder_dims, bias=False)
        self.decoder = Decoder(n_mels, decoder_dims, lstm_dims, speaker_latent_dims, noise_latent_dims)
        self.postnet = CBHG(postnet_K, n_mels + noise_latent_dims, postnet_dims, [256, n_mels + noise_latent_dims], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, fft_bins, bias=False)

        # VAE Domain Adversarial ########################################################
        if hp.encoder_model == "CNN":
            self.speaker_encoder = CNNEncoder(n_mels, speaker_latent_dims, speaker_encoder_dims)
            self.noise_encoder = CNNEncoder(n_mels, noise_latent_dims, noise_encoder_dims)
        elif hp.encoder_model == "CNNRNN":
            self.speaker_encoder = CNNRNNEncoder(n_mels, speaker_latent_dims, speaker_encoder_dims)
            self.noise_encoder = CNNRNNEncoder(n_mels, noise_latent_dims, noise_encoder_dims)

        self.speaker_speaker = Classifier(speaker_latent_dims, n_speakers)
        self.speaker_noise = Classifier(speaker_latent_dims, 2)
        self.noise_speaker = Classifier(noise_latent_dims, n_speakers)
        self.noise_noise = Classifier(noise_latent_dims, 2)
        ## speaker encoder prior
        self.speaker_latent_loc = nn.Parameter(torch.zeros(speaker_latent_dims), requires_grad=False)
        self.speaker_latent_scale = nn.Parameter(torch.ones(speaker_latent_dims), requires_grad=False)
        self.speaker_latent_prior = dist.Independent(dist.Normal(self.speaker_latent_loc, self.speaker_latent_scale), 1)
        ## noise encoder prior
        self.noise_latent_loc = nn.Parameter(torch.zeros(noise_latent_dims), requires_grad=False)
        self.noise_latent_scale = nn.Parameter(torch.ones(noise_latent_dims), requires_grad=False)
        self.noise_latent_prior = dist.Independent(dist.Normal(self.noise_latent_loc, self.noise_latent_scale), 1)

        #################################################################################

        self.init_model()
        self.num_params()
        self.register_buffer("step", torch.zeros(1).long())
        self.register_buffer("r", torch.tensor(0).long())

    def set_r(self, r):
        self.r.data = torch.tensor(r)
        self.decoder.r = r

    def get_r(self):
        return self.r.item()

    def forward(self, phone, mel, phone_len, frame_len):
        """
        :param phone: Long[Batch, Text]
        :param mel: [Batch, NMel, Time]
        :param phone_len: [Batch] LongTensor
        :param frame_len: [Batch] LongTensor for the length in frame
        :return:
        Predicted Mel [Batch, NMel, (Time + r - 1) // r * r]
        Predicted Linear [Batch, NFFT, (Time + r - 1) // r * r]
        Attention [Batch, (Time + r - 1) // r, Text]
        Speaker Prediction [Batch, Speaker]
        Augment Prediction [Batch, 2]
        KL Loss for VAE [Batch]
        """
        device = next(self.parameters()).device  # use same device as parameters

        self.step += 1

        batch_size, _, steps = mel.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device) # [Batch, NMels]

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device) # [Batch, Decoder]

        # this avoids unnecessary matmul in decoder loop
        encoder_seq = self.encoder(phone)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # VAE Domain Adversarial Training Area ########################################
        noise_loc, noise_log_scale = self.noise_encoder(mel, frame_len, "noise_conv", "noise_encode") # [Batch, Noise]
        speaker_loc, speaker_log_scale = self.speaker_encoder(mel, frame_len, "speaker_conv", "speaker_encode") # [Batch, Speaker]
        hp.hook["logs"] = speaker_log_scale
        speaker_dist = dist.Independent(dist.Normal(speaker_loc, torch.exp(speaker_log_scale)), 1)
        noise_dist = dist.Independent(dist.Normal(noise_loc, torch.exp(noise_log_scale)), 1)
        speaker_embedding = speaker_dist.rsample()
        noise_embedding = noise_dist.rsample()

        speaker_speaker_classification = self.speaker_speaker(speaker_embedding)
        speaker_noise_classification = self.speaker_noise(grad_reverse(speaker_embedding)) # [Batch, 2]
        noise_speaker_classification = self.noise_speaker(noise_embedding.detach())
        noise_noise_classification = self.noise_noise(noise_embedding.detach())

        kl_loss = dist.kl_divergence(speaker_dist, self.speaker_latent_prior) + dist.kl_divergence(noise_dist, self.noise_latent_prior)

        ###############################################################################
        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = mel[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(
                    encoder_seq,
                    encoder_seq_proj,
                    prenet_in,
                    hidden_states,
                    cell_states,
                    context_vec,
                    t,
                    phone_len,
                    speaker_embedding,
                    noise_embedding
                )
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2) # [Batch, NMel, floor(Time, r)]
        postnet_noise_embedding = noise_embedding.unsqueeze(-1).repeat(1, 1, steps)
        postnet_input = torch.cat([mel_outputs, postnet_noise_embedding], dim=1)
        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(postnet_input)
        linear_outputs = self.post_proj(postnet_out)
        linear_outputs = linear_outputs.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()

        return \
            mel_outputs, \
            linear_outputs, \
            speaker_speaker_classification, \
            speaker_noise_classification, \
            noise_speaker_classification, \
            noise_noise_classification, \
            kl_loss, \
            attn_scores

    def inference_speaker(self, mel):
        """
        Inference Speaker From Mel Input, this function accepts single Mel
        :param mel: [NMel, Time]
        :return: FloatTensor [Speaker]
        """
        device = self.device
        self.eval()
        # Forcing correct device input
        l = torch.as_tensor([len(mel)], dtype=torch.long, device=device)
        mel = torch.as_tensor(mel, dtype=torch.float, device=device)
        mel = mel.unsqueeze(0) # [1, NMel, Time]

        speaker_latent, _ = self.speaker_encoder(mel, l) # [1, Speaker]\
        speaker_latent = speaker_latent.squeeze(0)

        self.train()
        return speaker_latent

    def inference_noise(self, mel):
        """
        Inference Noise From Mel Input, this function accepts single Mel
        :param mel: [NMel, Time]
        :return: FloatTensor [Noise]
        """
        device = self.device
        self.eval()
        # Forcing correct device input
        l = torch.as_tensor([len(mel)], dtype=torch.long, device=device)
        mel = torch.as_tensor(mel, dtype=torch.float, device=device)
        mel = mel.unsqueeze(0) # [1, NMel, Time]

        noise_latent, _ = self.noise_encoder(mel, l) # [1, Noise]
        noise_latent = noise_latent.squeeze(0)

        self.train()
        return noise_latent

    def batch_inference_speaker(self, mels, lengths):
        """
        Inference Speaker From
        :param mels: Any [Batch, NMels, Time]
        :param lengths: Any [Batch]
        :return: FloatTensor [Batch, Speaker]
        """
        device = self.device
        self.eval()
        l = torch.as_tensor(lengths, dtype=torch.long, device=device) # [Batch]
        mels = torch.as_tensor(mels, dtype=torch.float, device=device) # [Batch, NMels, Time]
        speaker_latent, _ = self.speaker_encoder(mels, l) # [Batch, Speaker]
        self.train()
        return speaker_latent

    def batch_inference_noise(self, mels, lengths):
        """
        Inference Speaker From
        :param mels: Any [Batch, NMels, Time]
        :param lengths: Any [Batch]
        :return: FloatTensor [Batch, Noise]
        """
        device = self.device
        self.eval()
        l = torch.as_tensor(lengths, dtype=torch.long, device=device) # [Batch]
        mels = torch.as_tensor(mels, dtype=torch.float, device=device) # [Batch, NMels, Time]
        noise_latent, _ = self.noise_encoder(mels, l) # [Batch, Speaker]
        self.train()
        return noise_latent

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, phone, speaker_embedding, noise_embedding, max_steps=2000):
        """
        Notice that this function only supports inference on single sentence.
        :param phone: list of phone ids [int]
        :param speaker_embedding: FloatTensor [Speaker]
        :param noise_embedding: FloatTensor [Noise]
        :param max_steps: int for the maximum inference step.
        :return: []
        """
        # Forcing evaluation mode on Tacotron Model for Inference
        device = self.device
        self.eval()
        # Synthesis generation input
        batch_size = 1
        phone = torch.as_tensor(phone, dtype=torch.long, device=device).unsqueeze(0)
        phone_len = torch.as_tensor([phone.size(1)], dtype=torch.long, device=device)
        speaker_embedding = torch.as_tensor(speaker_embedding, dtype=torch.float, device=device).unsqueeze(0)
        noise_embedding = torch.as_tensor(noise_embedding, dtype=torch.float, device=device).unsqueeze(0)

        # Need to initialise all hidden states and pack into tuple for tidyness
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Need to initialise all lstm cell states and pack into tuple for tidyness
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # Need a <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(phone)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        # Need a couple of lists for outputs
        mel_outputs, attn_scores = [], []

        # Run the decoder loop
        for t in range(0, max_steps, self.r):
            prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(
                    encoder_seq,
                    encoder_seq_proj,
                    prenet_in,
                    hidden_states,
                    cell_states,
                    context_vec,
                    t,
                    phone_len,
                    speaker_embedding,
                    noise_embedding
                )
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)
            # Stop the loop if silent frames present
            if (scores[0, 0, -1] > 0.8) and t > 10: break

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)
        postnet_noise_embedding = noise_embedding.unsqueeze(-1).repeat(1, 1, mel_outputs.size(-1))
        postnet_input = torch.cat([mel_outputs, postnet_noise_embedding], dim=1)
        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(postnet_input)
        linear_outputs = self.post_proj(postnet_out)
        linear_outputs = linear_outputs.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        attn_scores = attn_scores.cpu().data.numpy()[0]

        self.train()
        return mel_outputs, linear_outputs, attn_scores

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def checkpoint(self, path):
        k_steps = self.get_step() // 1000
        self.save(f'{path}/checkpoint_{k_steps}k.pyt')

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def restore(self, path):
        if not os.path.exists(path):
            print('\nNew Tacotron Training Session...\n')
            # self.save(path)
        else:
            print(f'\nLoading Weights: "{path}"\n')
            self.load(path)
            self.decoder.r = self.r.item()

    def load(self, path, device='cpu'):
        # because PyTorch places on CPU by default, we follow those semantics by using CPU as default.
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
