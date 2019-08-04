from pathlib import Path
from os.path import expanduser
from os import makedirs


# CONFIG

model_id = "baseline"

## Audio Signal Processing

sampling_rate = 16000
n_fft = 2048
fft_bins = n_fft // 2 + 1
n_mels = 80
hop_length = int(sampling_rate * 12.5 // 1000)
win_length = int(sampling_rate * 60 // 1000) # 25 ~ 128
min_f = 40
max_f = 8000

## Dataset

wav_path = Path(expanduser("~/datasets/ljspeech/waves_16000/"))
text_path = Path(expanduser("~/datasets/ljspeech/metadata.csv"))

n_loading_threads = 4
n_blocking_threads = 1
## Logging Paths

decay = 0.99
root_path = Path(__file__).parent
training_log_path = root_path / "log" / "log.txt"
checkpoint_path = root_path / "log" / "checkpoint"
makedirs(str(checkpoint_path), exist_ok=True)
attention_plot_path = root_path / "log" / "attention"
makedirs(str(attention_plot_path), exist_ok=True)
mel_plot_path = root_path / "log" / "mel"
makedirs(str(mel_plot_path), exist_ok=True)

## Network Parameters

embed_dims = 256
encoder_dims = 128
decoder_dims = 256
assert encoder_dims * 2 == decoder_dims, "Mismatch dimensions"
postnet_dims = 128
encoder_K = 16
lstm_dims = 512
postnet_K = 8
num_highways = 4
dropout = 0.5
cleaner_names = ['english_cleaners']

## Training Parameters

device = "cuda"

### Make sure that they are multiples of 16
schedule = [
    (7,  1e-3,  10_000,  32),   # progressive training schedule
    (5,  1e-4, 100_000,  32),   # (r, lr, step, batch_size)
    (2,  1e-4, 180_000,  16)
]


max_frame_len = None            # if you have a couple of extremely long spectrograms you might want to use this
bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
checkpoint_interval = 2000        # checkpoints the model every X steps
plot_interval = 200
### Binning Loading
q_size = 240
redundancy = 6
load_weight_path = checkpoint_path / "checkpoint.pt"
