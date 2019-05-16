import torch

def scale(tensor, factor):
    # type: (Tensor, int) -> Tensor
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float32)

    return tensor / factor

def pad_trim(tensor, ch_dim, max_len, len_dim, fill_value):
    # type: (Tensor, int, int, int, float) -> Tensor
    assert tensor.size(ch_dim) < 128, \
        "Too many channels ({}) detected, see channels_first param.".format(tensor.size(ch_dim))
    if max_len > tensor.size(len_dim):
        padding = [max_len - tensor.size(len_dim)
                   if (i % 2 == 1) and (i // 2 != len_dim)
                   else 0
                   for i in range(4)]
        with torch.no_grad():
            tensor = torch.nn.functional.pad(tensor, padding, "constant", fill_value)
    elif max_len < tensor.size(len_dim):
        tensor = tensor.narrow(len_dim, 0, max_len)
    return tensor

def downmix_mono(tensor, ch_dim):
    # type: (Tensor, int) -> Tensor
    if not tensor.dtype.is_floating_point:
        tensor = tensor.to(torch.float32)

    tensor = torch.mean(tensor, ch_dim, True)
    return tensor

def lc2cl(tensor):
    # type: (Tensor) -> Tensor
    return tensor.transpose(0, 1).contiguous()

def spectrogram(sig, pad, window, n_fft, hop, ws, power, normalize):
    # type: (Tensor, int, Tensor, int, int, int, int, bool) -> Tensor
    assert sig.dim() == 2

    if pad > 0:
        with torch.no_grad():
            sig = torch.nn.functional.pad(sig, (pad, pad), "constant")
    window = window.to(sig.device)

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(sig, n_fft, hop, ws,
                        window, center=True,
                        normalized=False, onesided=True,
                        pad_mode='reflect').transpose(1, 2)
    if normalize:
        spec_f /= window.pow(2).sum().sqrt()
    spec_f = spec_f.pow(power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
    return spec_f
