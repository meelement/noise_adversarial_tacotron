import torch
from torch import optim
import torch.nn.functional as F
from utils.display import *
from audio.spec.mel import MelSTFT
import hp
from utils.text.symbols import symbols
from tacotron import Tacotron
from dataset import DataLoader


def np_now(x): return x.detach().cpu().numpy()


def train_loop(stft, model, optimizer, train_set, lr, train_steps):
    for p in optimizer.param_groups: p['lr'] = lr

    start = time.time()
    avg_loss = 0.0

    for i in range(1, train_steps + 1):
        Phone, Wave, PhoneLength, WaveLength, FrameLength = train_set.get_batch()
        optimizer.zero_grad()
        Mel, Linear = stft.transform(Wave)
        Melest, Linearest, AttentionWeight = model(Phone, Mel, PhoneLength)

        mel_loss = F.l1_loss(Melest, Mel)
        linear_loss = F.l1_loss(Linearest, Linear)
        loss = mel_loss + linear_loss
        #


        # Decay Loss
        if avg_loss:
            avg_loss = avg_loss * hp.decay + loss.item() * (1 - hp.decay)
        else:
            avg_loss = loss.item()

        loss.backward()

        if hp.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad_norm)

        optimizer.step()

        step = model.get_step()

        speed = i / (time.time() - start)

        if step % hp.checkpoint_interval == 0:
            model.checkpoint(hp.checkpoint_path)

        if step % hp.plot_interval == 0:
            save_attention(AttentionWeight[0][:, :160], str(hp.attention_plot_path / str(step)))
            save_spectrogram(np_now(Linearest[0]), str(hp.mel_plot_path / str(step)), 600)
        train_set_state = train_set.get_state()
        msg = f'|({i}/{train_steps}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {step // 1000}k | Pipe: {train_set_state[0]} / {train_set_state[1]}'
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
        dropout=hp.dropout
    ).to(device=hp.device)

    model.restore(str(hp.load_weight_path))
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
        wave_path=hp.wav_path,
        txt_path=hp.text_path,
        q_size=hp.q_size,
        n_loading_threads=hp.n_loading_threads,
        stft=stft,
        redundancy=hp.redundancy,
        device=hp.device
    )
    # Iterate on training session number:
    for session in hp.schedule:
        r, lr, max_step, batch_size = session
        if current_step < max_step:
            model.set_r(r)
            train_set.set_state(batch_size, r)
            training_steps = max_step - current_step
            simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                          ('Batch Size', batch_size),
                          ('Learning Rate', lr),
                          ('Outputs/Step (r)', model.get_r())])

            train_loop(stft, model, optimizer, train_set, lr, training_steps)
