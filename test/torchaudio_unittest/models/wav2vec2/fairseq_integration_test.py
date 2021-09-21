import json

import torch
from torchaudio.models.wav2vec2 import (
    wav2vec2_base,
    wav2vec2_large,
    wav2vec2_large_lv60k,
)
from torchaudio.models.wav2vec2.utils import (
    import_fairseq_model,
)
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    get_asset_path,
    skipIfNoModule,
    TorchaudioTestCase,
)


def _load_config(*paths):
    with open(f'{get_asset_path("wav2vec2", "fairseq", *paths)}.json', 'r') as file_:
        return json.load(file_)


def _name_func(testcase_func, i, param):
    return f'{testcase_func.__name__}_{i}_{param[0][1].__name__}'


# Pretrined (not fine-tuned) models
BASE = _load_config('wav2vec_small')
LARGE = _load_config('libri960_big')
LARGE_LV60K = _load_config('wav2vec_vox_new')
XLSR_53_56K = _load_config('xlsr_53_56k')
# Fine-tuned models
BASE_960H = _load_config('wav2vec_small_960h')
LARGE_960H = _load_config('wav2vec_large_960h')
LARGE_LV60K_960H = _load_config('wav2vec_large_lv60k_960h')
LARGE_LV60K_SELF_960H = _load_config('wav2vec_large_lv60k_self_960h')

# Config and corresponding factory functions
PRETRAINED_CONFIGS = parameterized.expand([
    (BASE, wav2vec2_base),
    (LARGE, wav2vec2_large),
    (LARGE_LV60K, wav2vec2_large_lv60k),
    (XLSR_53_56K, wav2vec2_large_lv60k),
], name_func=_name_func)
FINETUNED_CONFIGS = parameterized.expand([
    (BASE_960H, wav2vec2_base),
    (LARGE_960H, wav2vec2_large),
    (LARGE_LV60K_960H, wav2vec2_large_lv60k),
    (LARGE_LV60K_SELF_960H, wav2vec2_large_lv60k),
], name_func=_name_func)


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
        from fairseq.models.wav2vec.wav2vec2 import (
            Wav2Vec2Config,
            Wav2Vec2Model,
        )
        from fairseq.models.wav2vec.wav2vec2_asr import (
            Wav2VecEncoder,
            Wav2Vec2CtcConfig,
        )

        if config['_name'] == 'wav2vec_ctc':
            config = copy.deepcopy(config)
            config['w2v_args'] = OmegaConf.create(config['w2v_args'])
            return Wav2VecEncoder(Wav2Vec2CtcConfig(**config), num_out)
        if config['_name'] == 'wav2vec2':
            return Wav2Vec2Model(Wav2Vec2Config(**config))
        raise ValueError(f'Unexpected configuration: {config["_name"]}')

    @PRETRAINED_CONFIGS
    def test_import_pretrained_model(self, config, _):
        """Pretrained wav2vec2 models from fairseq can be imported and yields the same results"""
        num_out = 28
        batch_size, num_frames = 3, 1024

        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_model(original, 28).eval()

        x = torch.randn(batch_size, num_frames)
        hyp, _ = imported.extract_features(x)
        refs = original.extract_features(x, padding_mask=torch.zeros_like(x), layer=-1)
        for i, (ref, _) in enumerate(refs['layer_results']):
            self.assertEqual(hyp[i], ref.transpose(0, 1))

    @PRETRAINED_CONFIGS
    def test_recreate_pretrained_model(self, config, factory_func):
        """Imported pretrained models can be recreated via a factory function without fairseq."""
        num_out = 28
        batch_size, num_frames = 3, 1024

        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_model(original, 28).eval()

        reloaded = factory_func(num_out=num_out)
        reloaded.load_state_dict(imported.state_dict())
        reloaded.eval()

        x = torch.randn(batch_size, num_frames)
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])
        # Without mask
        ref, _ = imported(x)
        hyp, _ = reloaded(x)
        self.assertEqual(ref, hyp)

        # With mask
        ref, ref_lengths = imported(x, lengths)
        hyp, hyp_lengths = reloaded(x, lengths)
        self.assertEqual(ref, hyp)
        self.assertEqual(ref_lengths, hyp_lengths)

    @FINETUNED_CONFIGS
    def test_import_finetuned_model(self, config, _):
        """Fintuned wav2vec2 models from fairseq can be imported and yields the same results"""
        num_out = 28
        batch_size, num_frames = 3, 1024

        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_model(original).eval()

        # Without mask
        x = torch.randn(batch_size, num_frames)
        ref = original(x, torch.zeros_like(x))['encoder_out'].transpose(0, 1)
        hyp, _ = imported(x)
        self.assertEqual(ref, hyp)

        # With mask
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])
        mask = torch.arange(num_frames).expand(batch_size, num_frames) >= lengths[:, None]
        ref = original(x, mask)['encoder_out'].transpose(0, 1)
        hyp, output_lengths = imported(x, lengths)
        for i, l in enumerate(output_lengths):
            self.assertEqual(ref[i, :l, ...], hyp[i, :l, ...])

    @FINETUNED_CONFIGS
    def test_recreate_finetuned_model(self, config, factory_func):
        """Imported finetuned models can be recreated via a factory function without fairseq."""
        num_out = 28
        batch_size, num_frames = 3, 1024

        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_model(original).eval()

        reloaded = factory_func(num_out=num_out)
        reloaded.load_state_dict(imported.state_dict())
        reloaded.eval()

        # Without mask
        torch.manual_seed(0)
        x = torch.randn(batch_size, num_frames)
        ref, _ = imported(x)
        hyp, _ = reloaded(x)
        self.assertEqual(ref, hyp)

        # With mask
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])
        ref, ref_lengths = imported(x, lengths)
        hyp, hyp_lengths = reloaded(x, lengths)
        self.assertEqual(ref, hyp)
        self.assertEqual(ref_lengths, hyp_lengths)
