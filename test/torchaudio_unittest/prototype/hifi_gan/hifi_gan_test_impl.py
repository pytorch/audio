import importlib
import math
import os
import subprocess
import sys

import torch
from torchaudio.prototype.models import hifigan_model
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class HiFiGANTestImpl(TestBaseMixin):
    def _get_input_config(self):
        model_config = self._get_model_config()
        return {
            "batch_size": 7,
            "in_channels": model_config["in_channels"],
            "time_length": 10,
        }

    def _get_model_config(self):
        return {
            "upsample_rates": (8, 8, 4),
            "upsample_kernel_sizes": (16, 16, 8),
            "upsample_initial_channel": 256,
            "resblock_kernel_sizes": (3, 5, 7),
            "resblock_dilation_sizes": ((1, 2), (2, 6), (3, 12)),
            "resblock_type": 2,
            "in_channels": 80,
            "lrelu_slope": 0.1,
        }

    def _get_model(self):
        return hifigan_model(**self._get_model_config()).to(device=self.device, dtype=self.dtype).eval()

    def _get_inputs(self):
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        time_length = input_config["time_length"]
        in_channels = input_config["in_channels"]

        input = torch.rand(batch_size, in_channels, time_length).to(device=self.device, dtype=self.dtype)
        return input

    def _import_original_impl(self):
        """Clone the original implmentation and import necessary objects. Used in a test below checking that output
        of our implementation matches the original one.
        """
        module_name = "hifigan_cloned"
        path_cloned = "/tmp/" + module_name
        if not os.path.isdir(path_cloned):
            subprocess.run(["git", "clone", "https://github.com/jik876/hifi-gan.git", path_cloned])
            subprocess.run(["git", "checkout", "4769534d45265d52a904b850da5a622601885777"], cwd=path_cloned)
        sys.path.append("/tmp")
        sys.path.append(path_cloned)
        env = importlib.import_module(module_name + ".env")
        models = importlib.import_module(module_name + ".models")
        return env.AttrDict, models.Generator

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(31)

    def test_torchscript_consistency_forward(self):
        r"""Verify that scripting the model does not change the behavior of method `forward`."""
        inputs = self._get_inputs()

        original_model = self._get_model()
        scripted_model = torch_script(original_model).eval()

        for _ in range(2):
            ref_out = original_model(inputs)
            scripted_out = scripted_model(inputs)
            self.assertEqual(ref_out, scripted_out)

    def test_output_shape_forward(self):
        r"""Check that method `forward` produces correctly-shaped outputs."""
        input_config = self._get_input_config()
        model_config = self._get_model_config()

        batch_size = input_config["batch_size"]
        time_length = input_config["time_length"]

        inputs = self._get_inputs()
        model = self._get_model()

        for _ in range(2):
            out = model(inputs)
            self.assertEqual(
                (batch_size, 1, math.prod(model_config["upsample_rates"]) * time_length),
                out.shape,
            )

    def test_original_implementation_match(self):
        r"""Check that output of our implementation matches the original one."""
        AttrDict, Generator = self._import_original_impl()

        model_config = self._get_model_config()
        model_config = AttrDict(model_config)
        model_config.resblock = "1" if model_config.resblock_type == 1 else "2"
        model_ref = Generator(model_config).to(device=self.device, dtype=self.dtype)
        model_ref.remove_weight_norm()

        inputs = self._get_inputs()
        model = self._get_model()
        model.load_state_dict(model_ref.state_dict())

        ref_output = model_ref(inputs)
        output = model(inputs)
        self.assertEqual(ref_output, output)
