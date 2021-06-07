#!/usr/bin/env python3
import argparse
import logging
import os

import torch
import torchaudio
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
import simple_ctc


_LG = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to the input pretrained weight file.'
    )
    parser.add_argument(
        '--output-path',
        help='Path to the directory, where the Torchscript-ed pipelines are saved.',
    )
    parser.add_argument(
        '--test-file',
        help='Path to a test audio file.',
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Quantize the model.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help=(
            'When enabled, individual components are separately tested '
            'for the numerical compatibility and TorchScript compatibility.'
        )
    )
    return parser.parse_args()


class Loader(torch.nn.Module):
    def forward(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, float(sample_rate), 16000.)
        return waveform


class Encoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        length = torch.tensor([waveform.shape[1]])
        result, length = self.encoder(waveform, length)
        return result


class Decoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, emission: torch.Tensor) -> str:
        result = self.decoder.decode(emission)
        return ''.join(result.label_sequences[0][0]).replace('|', ' ')


def _get_model(model_id):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    tokenizer = Wav2Vec2Processor.from_pretrained(model_id).tokenizer
    labels = [k for k, v in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])]
    original = Wav2Vec2ForCTC.from_pretrained(model_id)
    model = import_huggingface_model(original)
    return model.eval(), labels


def _get_decoder(labels):
    return Decoder(
        simple_ctc.BeamSearchDecoder(
            labels,
            cutoff_top_n=40,
            cutoff_prob=0.8,
            beam_size=100,
            num_processes=1,
            blank_id=0,
            is_nll=True,
        )
    )


def _main():
    args = _parse_args()
    _init_logging(args.debug)
    _LG.info('Loading model: %s', args.model)
    model, labels = _get_model(args.model)
    _LG.info('Labels: %s', labels)
    _LG.info('Building pipeline')
    loader = Loader()
    encoder = Encoder(model)
    decoder = _get_decoder(labels)
    _LG.info(encoder)

    if args.quantize:
        _LG.info('Quantizing the model')
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        encoder = torch.quantization.quantize_dynamic(
            encoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        _LG.info(encoder)

    # test
    if args.test_file:
        _LG.info('Testing with %s', args.test_file)
        waveform = loader(args.test_file)
        emission = encoder(waveform)
        transcript = decoder(emission)
        _LG.info(transcript)

    torch.jit.script(loader).save(os.path.join(args.output_path, 'loader.zip'))
    torch.jit.script(encoder).save(os.path.join(args.output_path, 'encoder.zip'))
    torch.jit.script(decoder).save(os.path.join(args.output_path, 'decoder.zip'))


def _init_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    format_ = (
        '%(message)s' if not debug else
        '%(asctime)s: %(levelname)7s: %(funcName)10s: %(message)s'
    )
    logging.basicConfig(level=level, format=format_)


if __name__ == '__main__':
    _main()
