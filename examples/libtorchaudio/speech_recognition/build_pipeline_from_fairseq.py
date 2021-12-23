#!/usr/bin/evn python3
"""Build Speech Recognition pipeline based on fairseq's wav2vec2.0 and dump it to TorchScript file.

To use this script, you need `fairseq`.
"""
import argparse
import logging
import os
from typing import Tuple

import fairseq
import torch
import torchaudio
from greedy_decoder import Decoder
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchaudio.models.wav2vec2.utils.import_fairseq import import_fairseq_model

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 10):
    import torch.ao.quantization as tq
else:
    import torch.quantization as tq

_LG = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--model-file", required=True, help="Path to the input pretrained weight file.")
    parser.add_argument(
        "--dict-dir",
        help=(
            "Path to the directory in which `dict.ltr.txt` file is found. " "Required only when the model is finetuned."
        ),
    )
    parser.add_argument(
        "--output-path",
        help="Path to the directory, where the TorchScript-ed pipelines are saved.",
    )
    parser.add_argument(
        "--test-file",
        help="Path to a test audio file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "When enabled, individual components are separately tested "
            "for the numerical compatibility and TorchScript compatibility."
        ),
    )
    parser.add_argument("--quantize", action="store_true", help="Apply quantization to model.")
    parser.add_argument("--optimize-for-mobile", action="store_true", help="Apply optmization for mobile.")
    return parser.parse_args()


class Loader(torch.nn.Module):
    def forward(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, float(sample_rate), 16000.0)
        return waveform


class Encoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        result, _ = self.encoder(waveform)
        return result[0]


def _get_decoder():
    labels = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "|",
        "E",
        "T",
        "A",
        "O",
        "N",
        "I",
        "H",
        "S",
        "R",
        "D",
        "L",
        "U",
        "M",
        "W",
        "C",
        "F",
        "G",
        "Y",
        "P",
        "B",
        "V",
        "K",
        "'",
        "X",
        "J",
        "Q",
        "Z",
    ]
    return Decoder(labels)


def _load_fairseq_model(input_file, data_dir=None):
    overrides = {}
    if data_dir:
        overrides["data"] = data_dir

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([input_file], arg_overrides=overrides)
    model = model[0]
    return model


def _get_model(model_file, dict_dir):
    original = _load_fairseq_model(model_file, dict_dir)
    model = import_fairseq_model(original.w2v_encoder)
    return model


def _main():
    args = _parse_args()
    _init_logging(args.debug)
    loader = Loader()
    model = _get_model(args.model_file, args.dict_dir).eval()
    encoder = Encoder(model)
    decoder = _get_decoder()
    _LG.info(encoder)

    if args.quantize:
        _LG.info("Quantizing the model")
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        encoder = tq.quantize_dynamic(encoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        _LG.info(encoder)

    # test
    if args.test_file:
        _LG.info("Testing with %s", args.test_file)
        waveform = loader(args.test_file)
        emission = encoder(waveform)
        transcript = decoder(emission)
        _LG.info(transcript)

    torch.jit.script(loader).save(os.path.join(args.output_path, "loader.zip"))
    torch.jit.script(decoder).save(os.path.join(args.output_path, "decoder.zip"))
    scripted = torch.jit.script(encoder)
    if args.optimize_for_mobile:
        scripted = optimize_for_mobile(scripted)
    scripted.save(os.path.join(args.output_path, "encoder.zip"))


def _init_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    format_ = "%(message)s" if not debug else "%(asctime)s: %(levelname)7s: %(funcName)10s: %(message)s"
    logging.basicConfig(level=level, format=format_)


if __name__ == "__main__":
    _main()
