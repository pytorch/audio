import sys

import torch
from torchaudio.prototype.pipelines import HIFIGAN_GENERATOR_V3_LJSPEECH
from torchaudio_unittest.common_utils import download_file_from_google_drive, PytorchTestCase, skipIfNoModule

from .hifi_gan_test_impl import _import_hifi_gan_original_impl


class HiFiGANPretrainedTest(PytorchTestCase):
    """Test that HiFiGAN model can be created from the bundle and that the weights are the same as in the original
    publication. This test is implemented separately from HiFiGANTestImpl, because only needs to run on one device and
    one data type.
    """

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(31)
        # Import code necessary for test_original_implementation_match
        self.AttrDict, self.Generator = _import_hifi_gan_original_impl()
        self.bundle = HIFIGAN_GENERATOR_V3_LJSPEECH

    def tearDown(self):
        # PATH was modified on test setup, revert the modifications
        sys.path.pop(0)
        sys.path.pop(0)

    def test_smoke_bundle(self):
        """Smoke test of downloading weights for pretraining models"""
        self.bundle.get_vocoder()

    @skipIfNoModule("requests")
    def test_pretrained_weights(self):
        """Test that pre-trained weights from the original implementation saved on Google Drive
        match weights in our bundle.
        """
        weights_download_path = "/tmp/weights_v3.bin"
        # Download original V3 weights from Google Drive
        download_file_from_google_drive("18TNnHbr4IlduAWdLrKcZrqmbfPOed1pS", weights_download_path)

        # Instantiate the original model using parameters from the bundle
        model_ref = self.Generator
        model_config = self.AttrDict(self.bundle._params)
        model_config.resblock = "1" if model_config.resblock_type == 1 else "2"
        model_ref = self.Generator(model_config)
        # Load the original weights into the model
        loaded_model = torch.load(weights_download_path, map_location=torch.device("cpu"))
        model_ref.load_state_dict(loaded_model["generator"])
        model_ref.remove_weight_norm()

        # Check that the weights loaded from the original implementation and from our bundle are equal
        model_bundle = self.bundle.get_vocoder()
        self.assertEqual(model_ref.state_dict(), model_bundle.state_dict())
