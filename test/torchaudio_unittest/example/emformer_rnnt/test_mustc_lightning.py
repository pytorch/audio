from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import torch
from parameterized import parameterized
from torchaudio._internal.module_utils import is_module_available
from torchaudio_unittest.common_utils import TorchaudioTestCase, skipIfNoModule

from .utils import MockSentencePieceProcessor, MockCustomDataset

if is_module_available("pytorch_lightning", "sentencepiece"):
    from asr.emformer_rnnt.mustc.lightning import MuSTCRNNTModule


class MockMUSTC:
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, n: int):
        return (
            torch.rand(1, 32640),
            "sup",
        )

    def __len__(self):
        return 10


@contextmanager
def get_lightning_module():
    with patch("sentencepiece.SentencePieceProcessor", new=partial(MockSentencePieceProcessor, num_symbols=500)), patch(
        "asr.emformer_rnnt.mustc.lightning.GlobalStatsNormalization", new=torch.nn.Identity
    ), patch("asr.emformer_rnnt.mustc.lightning.MUSTC", new=MockMUSTC), patch(
        "asr.emformer_rnnt.mustc.lightning.CustomDataset", new=MockCustomDataset
    ):
        yield MuSTCRNNTModule(
            mustc_path="mustc_path",
            sp_model_path="sp_model_path",
            global_stats_path="global_stats_path",
        )


@skipIfNoModule("pytorch_lightning")
@skipIfNoModule("sentencepiece")
class TestMuSTCRNNTModule(TorchaudioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.random.manual_seed(31)

    @parameterized.expand(
        [
            ("train",),
            ("dev",),
            ("tst-COMMON",),
            ("tst-HE",),
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
            elif subset == "tst-COMMON":
                dataloader = lightning_module.test_common_dataloader()
                step = lightning_module.test_step
            elif subset == "tst-HE":
                dataloader = lightning_module.test_he_dataloader()
                step = lightning_module.test_step
            else:
                dataloader = lightning_module.val_dataloader()
                step = lightning_module
            batch = next(iter(dataloader))
            if subset == "forward":
                step(batch)
            else:
                step(batch, 0)
