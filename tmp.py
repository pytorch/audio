import torch
import torchaudio.functional as F

# torch.jit.script(F.istft)
tensor = torch.rand((1, 1000))
n_fft = 400
ws = 400
hop = 200
pad = 0
window = torch.hann_window(ws)
power = 2
normalize = False

print(F.spectrogram(tensor, pad, window, n_fft, hop, ws, power, normalize).size())

