import torch
from torch import optim
import torch.nn.functional as F
from utils.display import *
from audio.spec.mel import MelSTFT
import hp
from utils.text.symbols import symbols
from model.tacotron import Tacotron
from dataset import DataLoader


def np_now(x): return x.detach().cpu().numpy()


def print_all_grad(model: torch.nn.Module):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(f"{name:40s}: {parameter.grad.abs().mean().item()}")
        else:
            print(f"{name:40s}: None")


def train_loop(stft, model, optimizer, train_set, train_steps):

    start = time.time()
    avg_reg_loss = None
    avg_sn = None
    avg_ss = None
    avg_kl_loss = None
    avg_speaker_speaker_acc = None
    avg_speaker_noise_acc = None
    avg_noise_speaker_acc = None
    avg_noise_noise_acc = None

    for i in range(1, train_steps + 1):
        step = model.get_step()
        lr = hp.learning_rate(step)
        for p in optimizer.param_groups: p['lr'] = lr

        Phone, Wave, Speaker, Augmented, PhoneLength, WaveLength, FrameLength, R = train_set.get_batch()
        if R != model.get_r(): continue
        optimizer.zero_grad()
        Mel, Linear = stft.transform(Wave)
        mask = torch.arange(torch.max(FrameLength), device=Wave.device)[None, :] < FrameLength[:, None]
        mask = mask.unsqueeze(dim=1).float()

        Melest, Linearest, SpeakerSpeaker, SpeakerNoise, NoiseSpeaker, NoiseNoise, KLLoss, AttentionWeight = model(Phone, Mel, PhoneLength, FrameLength)
        mel_loss = F.l1_loss(Melest * mask, Mel)
        linear_loss = F.l1_loss(Linearest * mask, Linear)

        SpeakerSpeakerAcc = (torch.argmax(SpeakerSpeaker, dim=1) == Speaker).float().mean()
        SpeakerNoiseAcc = (torch.argmax(SpeakerNoise, dim=1) == Augmented).float().mean()
        NoiseSpeakerAcc = (torch.argmax(NoiseSpeaker, dim=1) == Speaker).float().mean()
        NoiseNoiseAcc = (torch.argmax(NoiseNoise, dim=1) == Augmented).float().mean()

        speaker_speaker_loss = hp.classification_ratio * F.cross_entropy(SpeakerSpeaker, Speaker)
        speaker_noise_loss = hp.classification_ratio * F.cross_entropy(SpeakerNoise, Augmented)
        noise_speaker_loss = hp.classification_ratio * F.cross_entropy(NoiseSpeaker, Speaker)
        noise_noise_loss = hp.classification_ratio * F.cross_entropy(NoiseNoise, Augmented)
        kl_loss = torch.mean(KLLoss)
        reg_loss = mel_loss + linear_loss
        classification_loss = speaker_speaker_loss + noise_speaker_loss + speaker_noise_loss + noise_noise_loss
        loss = \
            reg_loss + \
            hp.kl_loss_ratio(step) * kl_loss + \
            classification_loss
        # Decay Loss
        if avg_reg_loss:
            avg_reg_loss = avg_reg_loss * hp.decay + reg_loss.item() * (1 - hp.decay)
            avg_ss= avg_ss * hp.decay + speaker_speaker_loss.item() * (1 - hp.decay)
            avg_kl_loss  = avg_kl_loss * hp.decay + kl_loss.item()  * (1 - hp.decay)
            avg_sn = avg_sn * hp.decay + speaker_noise_loss.item() * (1 - hp.decay)
            avg_ns = avg_ns * hp.decay + noise_speaker_loss.item() * (1 - hp.decay)
            avg_nn = avg_nn * hp.decay + noise_noise_loss.item() * (1 - hp.decay)
            avg_speaker_speaker_acc = avg_speaker_speaker_acc * hp.decay + SpeakerSpeakerAcc.item() * (1 - hp.decay)
            avg_speaker_noise_acc = avg_speaker_noise_acc * hp.decay + SpeakerNoiseAcc.item() * (1 - hp.decay)
            avg_noise_speaker_acc = avg_noise_speaker_acc * hp.decay + NoiseSpeakerAcc.item() * (1 - hp.decay)
            avg_noise_noise_acc = avg_noise_noise_acc * hp.decay + NoiseNoiseAcc.item() * (1 - hp.decay)

        else:
            avg_reg_loss = reg_loss.item()
            avg_kl_loss = kl_loss.item()
            avg_ss = speaker_speaker_loss.item()
            avg_sn = speaker_noise_loss.item()
            avg_ns = noise_speaker_loss.item()
            avg_nn = noise_noise_loss.item()
            avg_speaker_speaker_acc = SpeakerSpeakerAcc.item()
            avg_speaker_noise_acc = SpeakerNoiseAcc.item()
            avg_noise_speaker_acc = NoiseSpeakerAcc.item()
            avg_noise_noise_acc = NoiseNoiseAcc.item()

        if hp.debug and False:
            speaker_noise_loss.backward()
            print_all_grad(model)
            input("System Paused. ")

        loss.backward()

        if hp.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad_norm)

        optimizer.step()

        speed = i / (time.time() - start)

        if step % hp.checkpoint_interval == 0:
            model.checkpoint(hp.checkpoint_path)

        if step % hp.plot_interval == 0:
            save_attention(AttentionWeight[0][:, :160], str(hp.attention_plot_path / str(step)))
            save_spectrogram(np_now(Melest[0]), str(hp.mel_plot_path / str(step)), 600)
            save_spectrogram(np_now(Linearest[0]), str(hp.linear_plot_path / str(step)), 600)
            save_imshow(np_now(hp.hook["speaker_conv"]), str(hp.speaker_hidden_plot_path / str(step)), figsize=(12, 6))
            save_imshow(np_now(hp.hook["speaker_encode"]), str(hp.speaker_encode_plot_path / str(step)), figsize=(12, 6))
            save_imshow(np_now(hp.hook["noise_conv"]), str(hp.noise_hidden_plot_path / str(step)), figsize=(12, 6))
            save_imshow(np_now(hp.hook["noise_encode"]), str(hp.noise_encode_plot_path / str(step)), figsize=(12, 6))
            save_imshow(np_now(hp.hook["logs"]), str(hp.speaker_encode_plot_path / (str(step) + "_log")), figsize=(12, 6))


        train_set_state = train_set.get_state()
        msg = f'|({i:07d}/{train_steps:07d})|' \
              f'R:{avg_reg_loss:#.4}|' \
            f'KL:{avg_kl_loss:#.4}|' \
            f'S:{speed:#01.2}|'\
            f'S:{step//1000:03d}k|' \
            f'P:{train_set_state[0]}|' \
            f'SS:{avg_ss:#02.4}={(avg_speaker_speaker_acc * 100):02.0f}%|' \
            f'SN:{avg_sn:#02.4}={(avg_speaker_noise_acc * 100):02.0f}%|' \
            f'NS:{avg_ns:#02.4}={(avg_noise_speaker_acc * 100):02.0f}%|' \
            f'NN:{avg_nn:#02.4}={(avg_noise_noise_acc * 100):02.0f}%|' \
            f'LR:{lr:6.6f}@{hp.kl_loss_ratio(step):10.10f}|'
        stream(msg)
        model.log(hp.training_log_path, msg)


if __name__ == "__main__":
    print('Using device:', hp.device)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate MelSTFT Extractor
    mel_calc = MelSTFT(hp.n_fft, hp.hop_length, hp.win_length, hp.n_mels, hp.sampling_rate, hp.min_f, hp.max_f)

    # Instantiate Tacotron Model
    model = Tacotron(
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
        speaker_encoder_dims=hp.speaker_encoder_dims,
        speaker_latent_dims=hp.speaker_latent_dims,
        n_speakers=hp.n_speakers,
        noise_encoder_dims=hp.noise_encoder_dims,
        noise_latent_dims=hp.noise_latent_dims
    ).to(device=hp.device)

    model.restore(str(hp.load_weight_file))
    optimizer = optim.Adam(model.parameters())

    current_step = model.get_step()

    stft = MelSTFT(
        filter_length=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        n_mel_channels=hp.n_mels,
        sampling_rate=hp.sampling_rate,
        mel_fmin=hp.min_f,
        mel_fmax=hp.max_f
    ).to(device=hp.device)

    train_set = DataLoader(
        q_size=hp.q_size,
        n_loading_threads=hp.n_loading_threads,
        stft=stft,
        redundancy=hp.redundancy,
        device=hp.device
    )

    # Iterate on training session number:
    for session in hp.schedule:
        r, max_step, batch_size = session
        if current_step < max_step:
            model.set_r(r)
            train_set.set_state(batch_size, r)
            training_steps = max_step - current_step
            simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                          ('Batch Size', batch_size),
                          ('Outputs/Step (r)', model.get_r())])

            train_loop(stft, model, optimizer, train_set, training_steps)

    print("\nTraining Finished, Quiting...\n")
    exit(0)
