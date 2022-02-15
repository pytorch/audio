from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import torch
from torchaudio._internal.module_utils import is_module_available
from torchaudio_unittest.common_utils import TorchaudioTestCase, skipIfNoModule

from .utils import MockSentencePieceProcessor, MockCustomDataset

if is_module_available("pytorch_lightning", "sentencepiece"):
    from asr.emformer_rnnt.tedlium3.lightning import TEDLIUM3RNNTModule


class MockTEDLIUM:
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, n: int):
        return (
            torch.rand(1, 32640),
            16000,
            "sup",
            2,
            3,
            4,
        )

    def __len__(self):
        return 10


@contextmanager
def get_lightning_module():
    with patch("sentencepiece.SentencePieceProcessor", new=partial(MockSentencePieceProcessor, num_symbols=500)), patch(
        "asr.emformer_rnnt.tedlium3.lightning.GlobalStatsNormalization", new=torch.nn.Identity
    ), patch("torchaudio.datasets.TEDLIUM", new=MockTEDLIUM), patch(
        "asr.emformer_rnnt.tedlium3.lightning.CustomDataset", new=MockCustomDataset
    ):
        yield TEDLIUM3RNNTModule(
            tedlium_path="tedlium_path",
            sp_model_path="sp_model_path",
            global_stats_path="global_stats_path",
        )


@skipIfNoModule("pytorch_lightning")
@skipIfNoModule("sentencepiece")
class TestTEDLIUM3RNNTModule(TorchaudioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.random.manual_seed(31)

    def test_training_step(self):
        with get_lightning_module() as lightning_module:
            train_dataloader = lightning_module.train_dataloader()
            batch = next(iter(train_dataloader))
            lightning_module.training_step(batch, 0)

    def test_validation_step(self):
        with get_lightning_module() as lightning_module:
            val_dataloader = lightning_module.val_dataloader()
            batch = next(iter(val_dataloader))
            lightning_module.validation_step(batch, 0)

    def test_test_step(self):
        with get_lightning_module() as lightning_module:
            test_dataloader = lightning_module.test_dataloader()
            batch = next(iter(test_dataloader))
            lightning_module.test_step(batch, 0)

    def test_forward(self):
        with get_lightning_module() as lightning_module:
            val_dataloader = lightning_module.val_dataloader()
            batch = next(iter(val_dataloader))
            lightning_module(batch)
