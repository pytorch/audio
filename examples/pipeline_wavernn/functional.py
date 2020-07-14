import torch


def specgram_normalize(specgram, min_level_db):
    r"""Normalize the spectrogram with a minimum db value
    """

    specgram = 20 * torch.log10(torch.clamp(specgram, min=1e-5))
    return torch.clamp((min_level_db - specgram) / min_level_db, min=0, max=1)


def mulaw_encode(waveform, mu):
    r"""Waveform mulaw encoding
    """

    mu = mu - 1
    fx = (
        torch.sign(waveform)
        * torch.log(1 + mu * torch.abs(waveform))
        / torch.log(torch.as_tensor(1.0 + mu))
    )
    return torch.floor((fx + 1) / 2 * mu + 0.5).int()


def waveform_to_label(waveform, bits):
    r"""Transform waveform [-1, 1] to label [0, 2 ** bits - 1]
    """

    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.0) * (2 ** bits - 1) / 2
    return torch.clamp(waveform, 0, 2 ** bits - 1).int()


def label_to_waveform(label, bits):
    r"""Transform label [0, 2 ** bits - 1] to waveform [-1, 1]
    """

    return 2 * label / (2 ** bits - 1.0) - 1.0
