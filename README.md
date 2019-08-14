# NoiseAdversarialTacotron

## Introduction

This repository contains a reproduction of the following paper, which applied domain adversarial training and data augmentation to Tacotron 2 model using VCTK dataset. 

*Disentangling Correlated Speaker and Noise for Speech Synthesis via Data Augmentation and Adversarial Factorization* by Wei-Ning Hsu et al.

## TODOs

- The paper only experimented with additive noises, try to add a simple room simulator on it and verify the performance with convolutional noise. We need room impulse response to create such augmentation. 
- Try to train the speaker encoder with tons of speakers for one-shot voice cloning? This might be interesting.
- Add a Neural Vocoder to the model.

## Quick Start

- You need to have some noise dataset, like CHiME-4 background dataset.
- You need to have some TTS dataset, with hunderds of speakers.
  - If you are using VCTK, you need to either roughly cut VCTK or ASR align then cut it since the forward attention fall into local minimum and does not optimize when there are varied length long silence at the beginning of many training examples. This can be roughly done by removing silence by threshold.
  - I haven't tried with LibreSpeech, it should work.
- No preprocessing is required since we adopt Nvidia's code and do all feature extracion and data augmentation live on GPU. 
- Change the dataset configuration in `./hp.py` and all that in `./dataset/`. Following the same API design or rewrite the loader completely. This implementation uses a customized loader with two CPU pipelines and multi-theading. 

## Results

Currently the performance is slightly inferer than the demonstration released by the authors.

We trained on LJSpeech + VCTK instead of VCTK alone, and grapheme rather than phoneme is used to train the model. 

We implemented using Tacotron 1 rather than Tacotron 2. 

We haven't verify the case of correlated speaker and noise yet, currently with noise and speaker combining randomly at probability of 50%. 


At `140K` training steps and 24 hours on single Nvidia 1080Ti, the model already converged. Four classifiers are trained jointly with the model, two of them has gradient attached to the model and the others detached.  According to the paper, only adversarial training on speaker encoder is applied. 

We have no held out set now, the training time result is, this is not as good as reported result using LDA on test set from the paper.

||Speaker Encoder|Residual Encoder|
|:-|:-:|:-:|
|Speaker Classifier|100 %|13 % (10 % is LJSpeech)|
|Augment Classifier|50 %|84 %|

t-SNE visualization of latent gives very similar results to that on the paper. 

Synthesis on clear speech is reasonably good, but synthesis on noisy speech have occasionally failing alignments and large noise. 

To hear the results for yourself, [our result](http://liuzj.site:2333/reports/adv.html) and [google's result](https://google.github.io/tacotron/publications/adv_tts/index.html) can be found on the reports.

## Thanks

- [Nvidia Tacotron2](https://github.com/NVIDIA/tacotron2) for their implementation of STFT and Mel Binning on GPU, we borrowed the feature extraction code and found them efficient and does not form a bottleneck. 

- [fatchord's Tacotron1 Implementation](https://github.com/fatchord/WaveRNN) This implementation is fast in training and pleasant to read. We based this implementation on this repo's tacotron 1 implementation.

- [Forward Attention from J.X. Zhang from *Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis*](https://arxiv.org/abs/1807.06736) Forward Attention enabled the Tacotron 1 Model to have clear alignment with 1500 steps on LJSpeech and 5000 steps on VCTK Dataset. Thanks to @azraelkuan 's implementation of Forward Attention.
