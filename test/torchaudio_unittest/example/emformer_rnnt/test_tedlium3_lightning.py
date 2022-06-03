from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import torch
from parameterized import parameterized
from torchaudio._internal.module_utils import is_module_available
from torchaudio_unittest.common_utils import skipIfNoModule, TorchaudioTestCase

from .utils import MockCustomDataset, MockDataloader, MockSentencePieceProcessor

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
    ), patch(
        "torch.utils.data.DataLoader", new=MockDataloader
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
        super().setUpClass()

    @parameterized.expand(
        [
            ("training_step", "train_dataloader"),
            ("validation_step", "val_dataloader"),
            ("test_step", "test_dataloader"),
        ]
    )
    def test_step(self, step_fname, dataloader_fname):
        with get_lightning_module() as lightning_module:
            dataloader = getattr(lightning_module, dataloader_fname)()
            batch = next(iter(dataloader))
            getattr(lightning_module, step_fname)(batch, 0)

    @parameterized.expand(
        [
            ("val_dataloader",),
        ]
    )
    def test_forward(self, dataloader_fname):
        with get_lightning_module() as lightning_module:
            dataloader = getattr(lightning_module, dataloader_fname)()
            batch = next(iter(dataloader))
            lightning_module(batch)
