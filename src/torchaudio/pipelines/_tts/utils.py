import logging
import os

import torch
from torchaudio._internal import download_url_to_file, module_utils as _mod_utils


def _get_chars():
    return (
        "_",
        "-",
        "!",
        "'",
        "(",
        ")",
        ",",
        ".",
        ":",
        ";",
        "?",
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    )


def _get_phones():
    return (
        "_",
        "-",
        "!",
        "'",
        "(",
        ")",
        ",",
        ".",
        ":",
        ";",
        "?",
        " ",
        "AA",
        "AA0",
        "AA1",
        "AA2",
        "AE",
        "AE0",
        "AE1",
        "AE2",
        "AH",
        "AH0",
        "AH1",
        "AH2",
        "AO",
        "AO0",
        "AO1",
        "AO2",
        "AW",
        "AW0",
        "AW1",
        "AW2",
        "AY",
        "AY0",
        "AY1",
        "AY2",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "EH0",
        "EH1",
        "EH2",
        "ER",
        "ER0",
        "ER1",
        "ER2",
        "EY",
        "EY0",
        "EY1",
        "EY2",
        "F",
        "G",
        "HH",
        "IH",
        "IH0",
        "IH1",
        "IH2",
        "IY",
        "IY0",
        "IY1",
        "IY2",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OW0",
        "OW1",
        "OW2",
        "OY",
        "OY0",
        "OY1",
        "OY2",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UH0",
        "UH1",
        "UH2",
        "UW",
        "UW0",
        "UW1",
        "UW2",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    )


def _to_tensor(indices):
    lengths = torch.tensor([len(i) for i in indices], dtype=torch.int32)
    values = [torch.tensor(i) for i in indices]
    values = torch.nn.utils.rnn.pad_sequence(values, batch_first=True)
    return values, lengths


def _load_phonemizer(file, dl_kwargs):
    if not _mod_utils.is_module_available("dp"):
        raise RuntimeError("DeepPhonemizer is not installed. Please install it.")

    from dp.phonemizer import Phonemizer

    # By default, dp issues DEBUG level log.
    logger = logging.getLogger("dp")
    orig_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        url = f"https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/{file}"
        directory = os.path.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, file)
        if not os.path.exists(path):
            dl_kwargs = {} if dl_kwargs is None else dl_kwargs
            download_url_to_file(url, path, **dl_kwargs)
        return Phonemizer.from_checkpoint(path)
    finally:
        logger.setLevel(orig_level)


def _unnormalize_waveform(waveform: torch.Tensor, bits: int) -> torch.Tensor:
    r"""Transform waveform [-1, 1] to label [0, 2 ** bits - 1]"""
    waveform = torch.clamp(waveform, -1, 1)
    waveform = (waveform + 1.0) * (2**bits - 1) / 2
    return torch.clamp(waveform, 0, 2**bits - 1).int()


def _get_taco_params(n_symbols):
    return {
        "mask_padding": False,
        "n_mels": 80,
        "n_frames_per_step": 1,
        "symbol_embedding_dim": 512,
        "encoder_embedding_dim": 512,
        "encoder_n_convolution": 3,
        "encoder_kernel_size": 5,
        "decoder_rnn_dim": 1024,
        "decoder_max_step": 2000,
        "decoder_dropout": 0.1,
        "decoder_early_stopping": True,
        "attention_rnn_dim": 1024,
        "attention_hidden_dim": 128,
        "attention_location_n_filter": 32,
        "attention_location_kernel_size": 31,
        "attention_dropout": 0.1,
        "prenet_dim": 256,
        "postnet_n_convolution": 5,
        "postnet_kernel_size": 5,
        "postnet_embedding_dim": 512,
        "gate_threshold": 0.5,
        "n_symbol": n_symbols,
    }


def _get_wrnn_params():
    return {
        "upsample_scales": [5, 5, 11],
        "n_classes": 2**8,  # n_bits = 8
        "hop_length": 275,
        "n_res_block": 10,
        "n_rnn": 512,
        "n_fc": 512,
        "kernel_size": 5,
        "n_freq": 80,
        "n_hidden": 128,
        "n_output": 128,
    }
