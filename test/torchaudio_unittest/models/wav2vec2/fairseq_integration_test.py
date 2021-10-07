import json

import torch
from torchaudio.models.wav2vec2 import (
    wav2vec2_base,
    wav2vec2_large,
    wav2vec2_large_lv60k,
    hubert_base,
    hubert_large,
    hubert_xlarge,
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


# Pretraining models
WAV2VEC2_BASE = _load_config('wav2vec_small')
WAV2VEC2_LARGE = _load_config('libri960_big')
WAV2VEC2_LARGE_LV60K = _load_config('wav2vec_vox_new')
WAV2VEC2_XLSR_53_56K = _load_config('xlsr_53_56k')
HUBERT_BASE = _load_config('hubert_base_ls960')
HUBERT_LARGE_LL60K = _load_config('hubert_large_ll60k')
HUBERT_XLARGE_LL60K = _load_config('hubert_xtralarge_ll60k')
# Finetuning models
WAV2VEC2_BASE_960H = _load_config('wav2vec_small_960h')
WAV2VEC2_LARGE_960H = _load_config('wav2vec_large_960h')
WAV2VEC2_LARGE_LV60K_960H = _load_config('wav2vec_large_lv60k_960h')
WAV2VEC2_LARGE_LV60K_SELF_960H = _load_config('wav2vec_large_lv60k_self_960h')
HUBERT_LARGE = _load_config('hubert_large_ll60k_finetune_ls960')
HUBERT_XLARGE = _load_config('hubert_xtralarge_ll60k_finetune_ls960')


# Config and corresponding factory functions
WAV2VEC2_PRETRAINING_CONFIGS = parameterized.expand([
    (WAV2VEC2_BASE, wav2vec2_base),
    (WAV2VEC2_LARGE, wav2vec2_large),
    (WAV2VEC2_LARGE_LV60K, wav2vec2_large_lv60k),
    (WAV2VEC2_XLSR_53_56K, wav2vec2_large_lv60k),
], name_func=_name_func)
HUBERT_PRETRAINING_CONFIGS = parameterized.expand([
    (HUBERT_BASE, hubert_base),
    (HUBERT_LARGE_LL60K, hubert_large),
    (HUBERT_XLARGE_LL60K, hubert_xlarge),
], name_func=_name_func)
ALL_PRETRAINING_CONFIGS = parameterized.expand([
    (WAV2VEC2_BASE, wav2vec2_base),
    (WAV2VEC2_LARGE, wav2vec2_large),
    (WAV2VEC2_LARGE_LV60K, wav2vec2_large_lv60k),
    (WAV2VEC2_XLSR_53_56K, wav2vec2_large_lv60k),
    (HUBERT_BASE, hubert_base),
    (HUBERT_LARGE_LL60K, hubert_large),
    (HUBERT_XLARGE_LL60K, hubert_xlarge),
], name_func=_name_func)
FINETUNING_CONFIGS = parameterized.expand([
    (WAV2VEC2_BASE_960H, wav2vec2_base),
    (WAV2VEC2_LARGE_960H, wav2vec2_large),
    (WAV2VEC2_LARGE_LV60K_960H, wav2vec2_large_lv60k),
    (WAV2VEC2_LARGE_LV60K_SELF_960H, wav2vec2_large_lv60k),
    (HUBERT_LARGE, hubert_large),
    (HUBERT_XLARGE, hubert_xlarge),
], name_func=_name_func)


@skipIfNoModule('fairseq')
class TestFairseqIntegration(TorchaudioTestCase):
    """Test the process of importing the models from fairseq.

    Test methods in this test suite check the following things
    1. Models loaded with fairseq cane be imported.
    2. The same model can be recreated without fairseq.
    """
    def _get_model(self, config, num_out=None):
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
        from fairseq.models.hubert.hubert_asr import (
            HubertCtcConfig,
            HubertEncoder,
        )
        from fairseq.models.hubert.hubert import (
            HubertModel,
            HubertConfig,
        )
        from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig

        if config['_name'] == 'wav2vec_ctc':
            config = copy.deepcopy(config)
            config['w2v_args'] = OmegaConf.create(config['w2v_args'])
            return Wav2VecEncoder(Wav2Vec2CtcConfig(**config), num_out)
        if config['_name'] == 'wav2vec2':
            return Wav2Vec2Model(Wav2Vec2Config(**config))
        if config['_name'] == 'hubert_ctc':
            config = copy.deepcopy(config)
            config['w2v_args'] = OmegaConf.create(config['w2v_args'])
            ctc_cfg = HubertCtcConfig(**config)
            return HubertEncoder(ctc_cfg, tgt_dict=range(num_out))
        if config['_name'] == 'hubert':
            dicts = [list(range(i)) for i in config['num_classes']]
            return HubertModel(
                HubertConfig(**config['model']),
                HubertPretrainingConfig(**config['task']),
                dicts,
            )
        raise ValueError(f'Unexpected configuration: {config["_name"]}')

    @WAV2VEC2_PRETRAINING_CONFIGS
    def test_import_wave2vec2_pretraining_model(self, config, _):
        """Wav2vec2 pretraining models from fairseq can be imported and yields the same results"""
        batch_size, num_frames = 3, 1024

        torch.manual_seed(0)
        original = self._get_model(config).eval()
        imported = import_fairseq_model(original).eval()

        x = torch.randn(batch_size, num_frames)
        hyp, _ = imported.extract_features(x)
        refs = original.extract_features(x, padding_mask=torch.zeros_like(x), layer=-1)
        for i, (ref, _) in enumerate(refs['layer_results']):
            self.assertEqual(hyp[i], ref.transpose(0, 1))

    @HUBERT_PRETRAINING_CONFIGS
    def test_import_hubert_pretraining_model(self, config, factory_func):
        """HuBERT pretraining models from fairseq can be imported and yields the same results"""
        batch_size, num_frames = 3, 1024

        torch.manual_seed(0)
        original = self._get_model(config).eval()
        imported = import_fairseq_model(original).eval()

        x = torch.randn(batch_size, num_frames)
        mask = torch.zeros_like(x)
        hyp, _ = imported.extract_features(x)

        # check the last layer
        ref, _ = original.extract_features(x, padding_mask=mask, output_layer=len(original.encoder.layers))
        atol = 3.0e-05 if factory_func is hubert_xlarge else 1.0e-5
        self.assertEqual(hyp[-1], ref, atol=atol, rtol=1.3e-6)

        # check the first layer
        ref, _ = original.extract_features(x, padding_mask=mask, output_layer=1)
        self.assertEqual(hyp[0], ref)

    @ALL_PRETRAINING_CONFIGS
    def test_recreate_pretraining_model(self, config, factory_func):
        """Imported pretraining models can be recreated via a factory function without fairseq."""
        batch_size, num_frames = 3, 1024

        original = self._get_model(config).eval()
        imported = import_fairseq_model(original).eval()

        reloaded = factory_func()
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

    @FINETUNING_CONFIGS
    def test_import_finetuning_model(self, config, _):
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

    @FINETUNING_CONFIGS
    def test_recreate_finetuning_model(self, config, factory_func):
        """Imported finetuning models can be recreated via a factory function without fairseq."""
        num_out = 28
        batch_size, num_frames = 3, 1024

        original = self._get_model(config, num_out).eval()
        imported = import_fairseq_model(original).eval()

        reloaded = factory_func(aux_num_out=num_out)
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
