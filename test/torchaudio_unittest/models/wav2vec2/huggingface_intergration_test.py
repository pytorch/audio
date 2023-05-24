import json

import torch
from parameterized import parameterized
from torchaudio.models.wav2vec2 import (
    wav2vec2_base,
    wav2vec2_large,
    wav2vec2_large_lv60k,
    wav2vec2_xlsr_1b,
    wav2vec2_xlsr_2b,
    wav2vec2_xlsr_300m,
    wavlm_base,
    wavlm_large,
)
from torchaudio.models.wav2vec2.utils import import_huggingface_model
from torchaudio_unittest.common_utils import (
    get_asset_path,
    skipIfCudaSmallMemory,
    skipIfNoModule,
    TorchaudioTestCase,
    zip_equal,
)


def _load_config(*paths):
    with open(f'{get_asset_path("wav2vec2", "huggingface", *paths)}.json', "r") as file_:
        return json.load(file_)


def _name_func(testcase_func, i, param):
    return f"{testcase_func.__name__}_{i}_{param[0][1].__name__}"


# Pretrained
HF_BASE = _load_config("wav2vec2-base")
HF_LARGE = _load_config("wav2vec2-large")
HF_LARGE_LV60 = _load_config("wav2vec2-large-lv60")
HF_LARGE_XLSR_53 = _load_config("wav2vec2-large-xlsr-53")
HF_BASE_10K_VOXPOPULI = _load_config("wav2vec2-base-10k-voxpopuli")
HF_BASE_WAVLM = _load_config("wavlm-base")
HF_LARGE_WAVLM = _load_config("wavlm-large")
HF_XLSR_300M = _load_config("wav2vec2-xls-r-300m")
HF_XLSR_1B = _load_config("wav2vec2-xls-r-1b")
HF_XLSR_2B = _load_config("wav2vec2-xls-r-2b")
# Finetuned
HF_BASE_960H = _load_config("wav2vec2-base-960h")
HF_LARGE_960H = _load_config("wav2vec2-large-960h")
HF_LARGE_LV60_960H = _load_config("wav2vec2-large-960h-lv60")
HF_LARGE_LV60_SELF_960H = _load_config("wav2vec2-large-960h-lv60-self")
HF_LARGE_XLSR_DE = _load_config("wav2vec2-large-xlsr-53-german")

# Config and corresponding factory functions
PRETRAIN_CONFIGS = parameterized.expand(
    [
        (HF_BASE, wav2vec2_base),
        (HF_LARGE, wav2vec2_large),
        (HF_LARGE_LV60, wav2vec2_large_lv60k),
        (HF_LARGE_XLSR_53, wav2vec2_large_lv60k),
        (HF_BASE_10K_VOXPOPULI, wav2vec2_base),
    ],
    name_func=_name_func,
)
XLSR_PRETRAIN_CONFIGS = parameterized.expand(
    [
        (HF_XLSR_300M, wav2vec2_xlsr_300m),
        (HF_XLSR_1B, wav2vec2_xlsr_1b),
        (HF_XLSR_2B, wav2vec2_xlsr_2b),
    ],
    name_func=_name_func,
)
FINETUNE_CONFIGS = parameterized.expand(
    [
        (HF_BASE_960H, wav2vec2_base),
        (HF_LARGE_960H, wav2vec2_large),
        (HF_LARGE_LV60_960H, wav2vec2_large_lv60k),
        (HF_LARGE_LV60_SELF_960H, wav2vec2_large_lv60k),
        (HF_LARGE_XLSR_DE, wav2vec2_large_lv60k),
    ],
    name_func=_name_func,
)
WAVLM_CONFIGS = parameterized.expand(
    [
        (HF_BASE_WAVLM, wavlm_base),
        (HF_LARGE_WAVLM, wavlm_large),
    ],
    name_func=_name_func,
)


@skipIfNoModule("transformers")
class TestHFIntegration(TorchaudioTestCase):
    """Test the process of importing the models from Hugging Face Transformers

    Test methods in this test suite check the following things
    1. Models loaded with Hugging Face Transformers cane be imported.
    2. The same model can be recreated without Hugging Face Transformers.
    """

    def _get_model(self, config):
        # Helper function to avoid importing transformers on module scope.
        # Normally, we use `is_module_available` helper function to check if
        # the library is available, and import it on module scope if available.
        # However, somehow, once "transformers" is imported, `is_module_available`
        # starts to fail. Therefore, we defer importing "transformers" until
        # the actual tests are started.
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model, WavLMConfig, WavLMModel

        if config["architectures"] == ["Wav2Vec2Model"]:
            return Wav2Vec2Model(Wav2Vec2Config(**config))
        if config["architectures"] == ["Wav2Vec2ForCTC"]:
            return Wav2Vec2ForCTC(Wav2Vec2Config(**config))
        if config["architectures"] == ["WavLMModel"]:
            return WavLMModel(WavLMConfig(**config))
        raise ValueError(f'Unexpected arch: {config["architectures"]}')

    def _test_import_pretrain(self, original, imported, config):
        # FeatureExtractor
        x = torch.randn(3, 1024)
        ref = original.feature_extractor(x).transpose(1, 2)
        hyp, _ = imported.feature_extractor(x, None)
        self.assertEqual(ref, hyp)
        # Feature projection
        x = torch.randn(3, 10, config["conv_dim"][-1])
        ref = original.feature_projection(x)[0]
        hyp = imported.encoder.feature_projection(x)
        self.assertEqual(ref, hyp)
        # Convolutional Positional Encoder
        x = torch.randn(3, 256, config["hidden_size"])
        ref = original.encoder.pos_conv_embed(x)
        hyp = imported.encoder.transformer.pos_conv_embed(x)
        self.assertEqual(ref, hyp)
        # Encoder Transformer Layer
        for original_, imported_ in zip(original.encoder.layers, imported.encoder.transformer.layers):
            b, l, e = 16, 3, config["hidden_size"]
            x = torch.randn(b, l, e)
            mask = torch.randn(b, 1, l, l)
            (ref,) = original_(x, attention_mask=mask, output_attentions=False)
            hyp, _ = imported_(x, mask)  # Ignore returned position_bias, which is always None for Wav2Vec2 and HuBERT
            self.assertEqual(ref, hyp)
        # The whole Encoder Transformer
        b, l, e = 16, 3, config["hidden_size"]
        x = torch.randn(b, l, e)
        ref = original.encoder(x).last_hidden_state
        hyp = imported.encoder.transformer(x)
        self.assertEqual(ref, hyp)

        # Test get_intermediate_outputs method
        b, l, e = 16, 3, config["hidden_size"]
        x = torch.randn(b, l, e)
        ref = original.encoder(x, output_hidden_states=True).hidden_states
        hyp = imported.encoder.transformer.get_intermediate_outputs(x)
        for i in range(len(hyp)):
            self.assertEqual(ref[i + 1], hyp[i], atol=1e-4, rtol=0.001)

    def _test_import_finetune(self, original, imported, config):
        # Aux
        x = torch.randn(3, 10, config["hidden_size"])
        ref = original.lm_head(x)
        hyp = imported.aux(x)
        self.assertEqual(ref, hyp)
        # The whole model without mask
        batch_size, num_frames = 3, 1024
        x = torch.randn(batch_size, num_frames)
        ref = original(x).logits
        hyp, _ = imported(x)
        self.assertEqual(ref, hyp)

        # The whole model with mask
        batch_size, num_frames = 3, 1024
        x = torch.randn(batch_size, num_frames)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )
        mask = torch.arange(num_frames).expand(batch_size, num_frames) < lengths[:, None]

        ref = original(x, attention_mask=mask).logits
        hyp, output_lengths = imported(x, lengths)

        for i, l in enumerate(output_lengths):
            self.assertEqual(ref[i, :l, ...], hyp[i, :l, ...])

    @PRETRAIN_CONFIGS
    def test_import_pretrain(self, config, _):
        """wav2vec2 models from HF transformers can be imported and yields the same results"""
        original = self._get_model(config).eval()
        imported = import_huggingface_model(original).eval()
        self._test_import_pretrain(original, imported, config)

    @XLSR_PRETRAIN_CONFIGS
    @skipIfCudaSmallMemory
    def test_import_xlsr_pretrain(self, config, _):
        """XLS-R models from HF transformers can be imported and yields the same results"""
        original = self._get_model(config).eval()
        imported = import_huggingface_model(original).eval()
        self._test_import_pretrain(original, imported, config)

    @FINETUNE_CONFIGS
    def test_import_finetune(self, config, _):
        """wav2vec2 models from HF transformers can be imported and yields the same results"""
        original = self._get_model(config).eval()
        imported = import_huggingface_model(original).eval()
        self._test_import_pretrain(original.wav2vec2, imported, config)
        self._test_import_finetune(original, imported, config)

    @WAVLM_CONFIGS
    def test_import_pretrain_wavlm(self, config, _):
        """WavLM models from HF transformers can be imported and yield the same results"""
        original = self._get_model(config).eval()
        imported = import_huggingface_model(original).eval()

        # FeatureExtractor
        x = torch.randn(3, 1024)
        ref = original.feature_extractor(x).transpose(1, 2)
        hyp, _ = imported.feature_extractor(x, None)
        self.assertEqual(ref, hyp)
        # Feature projection
        x = torch.randn(3, 10, config["conv_dim"][-1])
        ref = original.feature_projection(x)[0]
        hyp = imported.encoder.feature_projection(x)
        self.assertEqual(ref, hyp)
        # Convolutional Positional Encoder
        x = torch.randn(3, 256, config["hidden_size"])
        ref = original.encoder.pos_conv_embed(x)
        hyp = imported.encoder.transformer.pos_conv_embed(x)
        self.assertEqual(ref, hyp)

        position_bias = None
        position_bias_imp = None
        assert len(original.encoder.layers) > 0
        for original_, imported_ in zip_equal(original.encoder.layers, imported.encoder.transformer.layers):
            b, l, e = 16, 3, config["hidden_size"]
            x = torch.randn(b, l, e)
            mask = torch.randn(b, l) > 0.5  # HF WaveLM model expects the mask to be binary
            # HF WaveLM model (original_) takes in "attention mask" but actually uses it as key padding mask:
            # https://github.com/huggingface/transformers/blob/b047472650cba259621549ac27b18fd2066ce18e/src/transformers/models/wavlm/modeling_wavlm.py#L495
            ref, position_bias = original_(x, attention_mask=mask, output_attentions=False, position_bias=position_bias)
            hyp, position_bias_imp = imported_(x, key_padding_mask=mask.ne(1), position_bias=position_bias_imp)
            # Masked-out elements are undefined in the output
            ref_filled = ref.masked_fill(~mask.unsqueeze(2), 0)
            hyp_filled = hyp.masked_fill(~mask.unsqueeze(2), 0)
            self.assertEqual(ref_filled, hyp_filled)

        # The whole Encoder Transformer
        b, l, e = 16, 3, config["hidden_size"]
        x = torch.randn(b, l, e)
        ref = original.encoder(x).last_hidden_state
        hyp = imported.encoder.transformer(x)
        self.assertEqual(ref, hyp)

        # Test get_intermediate_outputs method
        b, l, e = 16, 3, config["hidden_size"]
        x = torch.randn(b, l, e)
        ref = original.encoder(x, output_hidden_states=True).hidden_states
        hyp = imported.encoder.transformer.get_intermediate_outputs(x)
        for i in range(len(hyp)):
            self.assertEqual(ref[i + 1], hyp[i], atol=1e-4, rtol=0.001)

    def _test_recreate(self, imported, reloaded, config):
        # FeatureExtractor
        x = torch.randn(3, 1024)
        ref, _ = imported.feature_extractor(x, None)
        hyp, _ = reloaded.feature_extractor(x, None)
        self.assertEqual(ref, hyp)
        # Feature projection
        x = torch.randn(3, 10, config["conv_dim"][-1])
        ref = imported.encoder.feature_projection(x)
        hyp = reloaded.encoder.feature_projection(x)
        self.assertEqual(ref, hyp)
        # Convolutional Positional Encoder
        x = torch.randn(3, 256, config["hidden_size"])
        ref = imported.encoder.transformer.pos_conv_embed(x)
        hyp = reloaded.encoder.transformer.pos_conv_embed(x)
        self.assertEqual(ref, hyp)
        # Encoder Transformer Layer
        for imported_, reloaded_ in zip(imported.encoder.transformer.layers, reloaded.encoder.transformer.layers):
            b, l, e = 16, 3, config["hidden_size"]
            x = torch.randn(b, l, e)
            mask = torch.randn(b, 1, l, l)

            ref = imported_(x, mask)
            hyp = reloaded_(x, mask)
            self.assertEqual(ref, hyp)
        # The whole Encoder Transformer
        # TODO: Add mask pattern. Expected mask shapes and values are different.
        b, l, e = 16, 3, config["hidden_size"]
        x = torch.randn(b, l, e)
        mask = torch.randn(b, 1, l, l)
        ref = imported.encoder.transformer(x)
        hyp = reloaded.encoder.transformer(x)
        self.assertEqual(ref, hyp)
        # Aux
        if imported.aux is not None:
            x = torch.randn(3, 10, config["hidden_size"])
            ref = imported.aux(x)
            hyp = reloaded.aux(x)
            self.assertEqual(ref, hyp)
        # The whole model
        x = torch.randn(3, 1024)
        ref, _ = imported(x)
        hyp, _ = reloaded(x)
        self.assertEqual(ref, hyp)

    @PRETRAIN_CONFIGS
    def test_recreate_pretrain(self, config, factory_func):
        """Imported models can be recreated via a factory function without Hugging Face transformers."""
        imported = import_huggingface_model(self._get_model(config)).eval()
        reloaded = factory_func()
        reloaded.load_state_dict(imported.state_dict())
        reloaded.eval()
        self._test_recreate(imported, reloaded, config)

    @FINETUNE_CONFIGS
    def test_recreate_finetune(self, config, factory_func):
        """Imported models can be recreated via a factory function without Hugging Face transformers."""
        imported = import_huggingface_model(self._get_model(config)).eval()
        reloaded = factory_func(aux_num_out=imported.aux.out_features)
        reloaded.load_state_dict(imported.state_dict())
        reloaded.eval()
        self._test_recreate(imported, reloaded, config)

    @WAVLM_CONFIGS
    def test_recreate_wavlm(self, config, factory_func):
        """Imported models can be recreated via a factory function without Hugging Face transformers."""
        imported = import_huggingface_model(self._get_model(config)).eval()
        reloaded = factory_func()
        reloaded.load_state_dict(imported.state_dict())
        reloaded.eval()

        # FeatureExtractor
        x = torch.randn(3, 1024)
        ref, _ = imported.feature_extractor(x, None)
        hyp, _ = reloaded.feature_extractor(x, None)
        self.assertEqual(ref, hyp)
        # Feature projection
        x = torch.randn(3, 10, config["conv_dim"][-1])
        ref = imported.encoder.feature_projection(x)
        hyp = reloaded.encoder.feature_projection(x)
        self.assertEqual(ref, hyp)
        # Convolutional Positional Encoder
        x = torch.randn(3, 256, config["hidden_size"])
        ref = imported.encoder.transformer.pos_conv_embed(x)
        hyp = reloaded.encoder.transformer.pos_conv_embed(x)
        self.assertEqual(ref, hyp)
        # Encoder Transformer Layer
        position_bias_ref = None
        position_bias_hyp = None
        for imported_, reloaded_ in zip(imported.encoder.transformer.layers, reloaded.encoder.transformer.layers):
            b, l, e = 16, 3, config["hidden_size"]
            x = torch.randn(b, l, e)
            mask = torch.randn(b, l) > 0.5  # HugginFace WaveLM expects the mask to be binary
            ref, position_bias_ref = imported_(x, key_padding_mask=mask, position_bias=position_bias_ref)
            hyp, position_bias_hyp = reloaded_(x, key_padding_mask=mask, position_bias=position_bias_hyp)
            self.assertEqual(ref, hyp)
        # The whole Encoder Transformer
        # TODO: Add mask pattern. Expected mask shapes and values are different.
        b, l, e = 16, 3, config["hidden_size"]
        x = torch.randn(b, l, e)
        mask = torch.randn(b, 1, l, l)
        ref = imported.encoder.transformer(x)
        hyp = reloaded.encoder.transformer(x)
        self.assertEqual(ref, hyp)
        # The whole model
        x = torch.randn(3, 1024)
        ref, _ = imported(x)
        hyp, _ = reloaded(x)
        self.assertEqual(ref, hyp)
