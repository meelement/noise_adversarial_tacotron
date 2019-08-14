from pathlib import Path
from os.path import expanduser
from os import makedirs
from random import sample
from os import system
from math import exp
from platform import node


# Debug Trigger ###############################################################
## Set your debug trigger here.
debug = "sorcerer" in node()
if debug:
    print("Working Under DEBUG Mode ! ")
# GLOBAL HOOKS ################################################################
# you can hook global variables here to print or observe their value.
# This is easier than wiring them out of the model.
# This resembles tensorflows' GraphKeys

hook = {}

# CONFIG ######################################################################
# This model works on VCTK and LJSpeech together, the model samples data from both dataset.
# This is a baseline multi-speaker model.
encoder_model = "CNNRNN"
assert encoder_model in ("CNN", "CNNRNN"), "Unknown Encoder Model"

model_id = f"{encoder_model}_" + "".join(sample("0123456789qwertyuiopasdfghjklzxcvbnm", 3))

print("My ID is ", model_id)

## Audio Signal Processing ####################################################

sampling_rate = 16000
n_fft = 2048
fft_bins = n_fft // 2 + 1
n_mels = 128
hop_length = int(sampling_rate * 12.5 // 1000)
win_length = int(sampling_rate * 60 // 1000) # 25 ~ 128
min_f = 40
max_f = 8000

## Dataset ####################################################################

vctk_wav_path = Path(expanduser("~/datasets/vctk/cut16/"))
vctk_text_path = Path(expanduser("~/datasets/vctk/txt16/"))
ljs_wav_path = Path(expanduser("~/datasets/ljspeech/wavs_16000/"))
ljs_text_path = Path(expanduser("~/datasets/ljspeech/metadata.csv"))
whole_chime_path = Path(expanduser("~/datasets/chime/backgrounds/"))
part_chime_path = Path(expanduser("~/datasets/chime/segmented_backgrounds/"))

n_loading_threads = 6

## Logging Paths ##############################################################

# low pass exponential decay rate used for filtering all kind of loss
decay = 0.9

root_path = Path(__file__).parent
makedirs(str(root_path / "log"), exist_ok=True)
log_root = root_path / "log" / model_id

training_log_path = log_root / "log.txt"
checkpoint_path = log_root / "checkpoint"
attention_plot_path = log_root / "attention"
mel_plot_path = log_root / "mel"
linear_plot_path = log_root / "linear"
speaker_hidden_plot_path = log_root / "speaker_hidden"
noise_hidden_plot_path = log_root / "noise_hidden"
speaker_encode_plot_path = log_root / "speaker_encode"
noise_encode_plot_path = log_root / "noise_encode"

## Automatically creating paths
temporary_paths = [
    log_root,
    checkpoint_path
]
for name, value in list(globals().items()):
    if "plot_path" in name:
        temporary_paths.append(value)


for path in temporary_paths:
    path.mkdir()

## Copying current hyper-parameters to the log dir.
system(f"cp {__file__} {str(log_root)}")



## Network Parameters #########################################################

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



## Training Parameters ########################################################

device = "cuda"
schedule = [
    (5, 70000, 32),   # (r, lr, step, batch_size)
    (3, 180000, 16),
]



init_lr = 0.0007
warmup_steps = 4000

# LR Decay is necessary for multi-speaker model to converge fast.
def learning_rate(step):
    return 0.0004 if debug else init_lr * warmup_steps ** 0.5 * min(step * warmup_steps ** -1.5, (step + 1) ** -0.5)

min_sample_length = sampling_rate
max_sample_length = sampling_rate * 8
bin_lengths = True
clip_grad_norm = 1.0
checkpoint_interval = 2000
plot_interval = 50 if debug else 200

### VAE Training

max_kl_ratio = 0.00000001
annealing_offset = 10000.0
annealing_ratio = 5000.0
classification_ratio = 1.0
noise_augment_probability = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def kl_loss_ratio(step):
    return max_kl_ratio * sigmoid((step - annealing_offset) / annealing_ratio)


n_speakers = 200
speaker_encoder_dims = 256
speaker_latent_dims = 64
noise_encoder_dims = 256
noise_latent_dims = 8

enable_speaker_guide = True
enable_adversarial = True

### Binning Loading ###########################################################

q_size = 1000
redundancy = 10
load_weight_file = root_path / "log" / "restore.pyt"
