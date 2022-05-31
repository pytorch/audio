from typing import List

import torch
from parameterized import parameterized
from torchaudio import sox_effects
from torchaudio_unittest.common_utils import (
    get_sinusoid,
    save_wav,
    skipIfNoSox,
    TempDirMixin,
    torch_script,
    TorchaudioTestCase,
)

from .common import load_params


class SoxEffectTensorTransform(torch.nn.Module):
    effects: List[List[str]]

    def __init__(self, effects: List[List[str]], sample_rate: int, channels_first: bool):
        super().__init__()
        self.effects = effects
        self.sample_rate = sample_rate
        self.channels_first = channels_first

    def forward(self, tensor: torch.Tensor):
        return sox_effects.apply_effects_tensor(tensor, self.sample_rate, self.effects, self.channels_first)


class SoxEffectFileTransform(torch.nn.Module):
    effects: List[List[str]]
    channels_first: bool

    def __init__(self, effects: List[List[str]], channels_first: bool):
        super().__init__()
        self.effects = effects
        self.channels_first = channels_first

    def forward(self, path: str):
        return sox_effects.apply_effects_file(path, self.effects, self.channels_first)


@skipIfNoSox
class TestTorchScript(TempDirMixin, TorchaudioTestCase):
    @parameterized.expand(
        load_params("sox_effect_test_args.jsonl"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects_tensor(self, args):
        effects = args["effects"]
        channels_first = True
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)

        trans = SoxEffectTensorTransform(effects, input_sr, channels_first)

        trans = torch_script(trans)

        wav = get_sinusoid(
            frequency=800, sample_rate=input_sr, n_channels=num_channels, dtype="float32", channels_first=channels_first
        )
        found, sr_found = trans(wav)
        expected, sr_expected = sox_effects.apply_effects_tensor(wav, input_sr, effects, channels_first)

        assert sr_found == sr_expected
        self.assertEqual(expected, found)

    @parameterized.expand(
        load_params("sox_effect_test_args.jsonl"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects_file(self, args):
        effects = args["effects"]
        channels_first = True
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)

        trans = SoxEffectFileTransform(effects, channels_first)
        trans = torch_script(trans)

        path = self.get_temp_path("input.wav")
        wav = get_sinusoid(
            frequency=800, sample_rate=input_sr, n_channels=num_channels, dtype="float32", channels_first=channels_first
        )
        save_wav(path, wav, sample_rate=input_sr, channels_first=channels_first)

        found, sr_found = trans(path)
        expected, sr_expected = sox_effects.apply_effects_file(path, effects, channels_first)

        assert sr_found == sr_expected
        self.assertEqual(expected, found)
