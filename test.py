import os
import math
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as transforms
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

import IPython

# tensor = torch.rand((1, 22050 * 2))
# sr = 16000
# freq = 440
# volume = .3
# tensor = (torch.cos(2 * math.pi * torch.arange(0, 4 * sr).float() * freq / sr))
# tensor = tensor.unsqueeze_(0)
# input_path = os.path.join('./test/assets', 'sinewave.wav')
input_path = os.path.join('./test/assets', 'steam-train-whistle-daniel_simon.mp3')
tensor, sr = torchaudio.load(input_path)
tensor = tensor.mean(dim=0, keepdim=True)
print(tensor.size())

n_fft = 2048
spectrogram = transforms.Spectrogram(n_fft=n_fft, hop_length=n_fft//4)
# griffinlim = transforms.GriffinLim(n_fft=n_fft, hop_length=n_fft//4, length=tensor.size(-1))
melscale = transforms.MelScale(n_mels=256, sample_rate=sr, n_stft=n_fft//2+1)
inv_mel = transforms.InverseMelScale(n_mels=256, sample_rate=sr, n_stft=n_fft//2+1)

method = torch.jit.script(inv_mel) # .cuda()

spec = spectrogram(tensor)
mel_spec = melscale(spec)
spec_hat = method(mel_spec.cuda()).cpu()
# tensor_hat = griffinlim(spec_hat)

# 
# spec = transform(tensor)
# print(spec.size())
# no_phase = torch.stack((spec, torch.zeros(*spec.size())), dim=-1)
# no_phase = F.istft(no_phase, n_fft, hop_length=n_fft//4, length=tensor.size(-1))
# print(no_phase.size())
# tensor_hat = inverse(spec)
# print(tensor_hat.size())
# 
# print((tensor_hat-tensor).max())
# print(torch.allclose(tensor, tensor_hat, atol=5e-3))
# 
# matplotlib.use('TkAgg')
# plt.figure()
# plt.subplot(3, 1, 1)
# librosa.display.waveplot(tensor.squeeze(0).numpy(), sr=sr)
# plt.title("Original")
# plt.subplot(3, 1, 2)
# librosa.display.waveplot(no_phase.squeeze(0).numpy(), sr=sr)
# plt.title("iSTFT")
# plt.subplot(3, 1, 3)
# librosa.display.waveplot(tensor_hat.squeeze(0).numpy(), sr=sr)
# plt.title("GriffinLim")
# # plt.tight_layout()
# plt.show()

IPython.embed()
