from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import torch
from parameterized import parameterized
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
        super().setUpClass()
        torch.random.manual_seed(31)

    @parameterized.expand(
        [
            ("train",),
            ("dev",),
            ("test",),
            ("forward",),
        ]
    )
    def test_step(self, subset):
        with get_lightning_module() as lightning_module:
            if subset == "train":
                dataloader = lightning_module.train_dataloader()
                step = lightning_module.training_step
            elif subset == "dev":
                dataloader = lightning_module.val_dataloader()
                step = lightning_module.validation_step
            elif subset == "test":
                dataloader = lightning_module.test_dataloader()
                step = lightning_module.test_step
            else:
                dataloader = lightning_module.val_dataloader()
                step = lightning_module
            batch = next(iter(dataloader))
            if subset == "forward":
                step(batch)
            else:
                step(batch, 0)
