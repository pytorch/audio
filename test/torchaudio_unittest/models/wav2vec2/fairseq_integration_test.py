import json

import torch
from torchaudio.models.wav2vec2 import (
    wav2vec2_base,
    wav2vec2_large,
    wav2vec2_large_lv60k,
)
from torchaudio.models.wav2vec2.utils import import_fairseq_finetuned_model
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    get_asset_path,
    skipIfNoModule,
    TorchaudioTestCase,
)


def _load_config(*paths):
    with open(f'{get_asset_path("wav2vec2", "fairseq", *paths)}.json', 'r') as file_:
        return json.load(file_)


FAIRSEQ_BASE_960H = _load_config('wav2vec_small_960h')
FAIRSEQ_LARGE_960H = _load_config('wav2vec_large_960h')
FAIRSEQ_LARGE_LV60K_960H = _load_config('wav2vec_large_lv60k_960h')
FAIRSEQ_LARGE_LV60K_SELF_960H = _load_config('wav2vec_large_lv60k_self_960h')

# Config and corresponding factory functions
FAIRSEQ_CONFIGS = [
    (FAIRSEQ_BASE_960H, wav2vec2_base),
    (FAIRSEQ_LARGE_960H, wav2vec2_large),
    (FAIRSEQ_LARGE_LV60K_960H, wav2vec2_large_lv60k),
    (FAIRSEQ_LARGE_LV60K_SELF_960H, wav2vec2_large_lv60k),
]


@skipIfNoModule('fairseq')
class TestFairseqIntegration(TorchaudioTestCase):
    """Test the process of importing the models from fairseq.

    Test methods in this test suite check the following things
    1. Models loaded with fairseq cane be imported.
    2. The same model can be recreated without fairseq.
    """
    def _get_model(self, config, num_out):
        import copy
        from omegaconf import OmegaConf
        from fairseq.models.wav2vec.wav2vec2_asr import (
            Wav2VecEncoder,
            Wav2VecCtc,
            Wav2Vec2CtcConfig,
        )

        config = copy.deepcopy(config)
        config['model']['w2v_args'] = OmegaConf.create(config['model']['w2v_args'])
        cfg = Wav2Vec2CtcConfig(**config['model'])
        w2v_encoder = Wav2VecEncoder(cfg, num_out)
        return Wav2VecCtc(cfg, w2v_encoder)

    @parameterized.expand([conf[:1] for conf in FAIRSEQ_CONFIGS])
    def test_import(self, config):
        """wav2vec2 models from fairseq can be imported and yields the same results"""
        num_out = 28

        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_finetuned_model(original, config).eval()

        batch_size, num_frames = 3, 1024
        # Without mask
        x = torch.randn(batch_size, num_frames)
        ref = original.w2v_encoder(x, torch.zeros_like(x))['encoder_out'].transpose(0, 1)
        hyp, _ = imported(x)
        self.assertEqual(ref, hyp)

        # With mask
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])
        mask = torch.arange(num_frames).expand(batch_size, num_frames) >= lengths[:, None]
        ref = original.w2v_encoder(x, mask)['encoder_out'].transpose(0, 1)
        hyp, output_lengths = imported(x, lengths)
        for i, l in enumerate(output_lengths):
            self.assertEqual(ref[i, :l, ...], hyp[i, :l, ...])

    @parameterized.expand(FAIRSEQ_CONFIGS)
    def test_recreate(self, config, factory_func):
        """Imported models can be recreated via a factory function without fairseq."""
        num_out = 28
        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_finetuned_model(original, config).eval()

        reloaded = factory_func(num_out=num_out)
        reloaded.load_state_dict(imported.state_dict())
        reloaded.eval()

        torch.manual_seed(0)
        x = torch.randn(3, 1024)
        ref, _ = imported(x)
        hyp, _ = reloaded(x)
        self.assertEqual(ref, hyp)
