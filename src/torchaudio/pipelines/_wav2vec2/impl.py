from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from torch.nn import Module

from . import aligner, utils


__all__ = []  # type: ignore


@dataclass
class Wav2Vec2Bundle:
    """Data class that bundles associated information to use pretrained :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - Feature Extraction
        >>> import torchaudio
        >>>
        >>> bundle = torchaudio.pipelines.HUBERT_BASE
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 360M/360M [00:06<00:00, 60.6MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Extract acoustic features
        >>> features, _ = model.extract_features(waveform)
    """  # noqa: E501

    _path: str
    _params: Dict[str, Any]
    _sample_rate: float
    _normalize_waveform: bool
    _model_type: str

    @property
    def sample_rate(self) -> float:
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate

    def _get_state_dict(self, dl_kwargs):
        # Note: This method is overridden in ASR bundle
        return utils._get_state_dict(self._path, dl_kwargs)

    def get_model(self, *, dl_kwargs=None) -> Module:
        """Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.models.Wav2Vec2Model`.

            For the models listed below, an additional layer normalization is performed on the input.

            For all other models, a :py:class:`~torchaudio.models.Wav2Vec2Model` instance is returned.

            - WAV2VEC2_LARGE_LV60K
            - WAV2VEC2_ASR_LARGE_LV60K_10M
            - WAV2VEC2_ASR_LARGE_LV60K_100H
            - WAV2VEC2_ASR_LARGE_LV60K_960H
            - WAV2VEC2_XLSR53
            - WAV2VEC2_XLSR_300M
            - WAV2VEC2_XLSR_1B
            - WAV2VEC2_XLSR_2B
            - HUBERT_LARGE
            - HUBERT_XLARGE
            - HUBERT_ASR_LARGE
            - HUBERT_ASR_XLARGE
            - WAVLM_LARGE
        """
        model = utils._get_model(self._model_type, self._params)
        state_dict = self._get_state_dict(dl_kwargs)
        model.load_state_dict(state_dict)
        if self._normalize_waveform:
            model = utils._extend_model(model, normalize_waveform=True)
        model.eval()
        return model


@dataclass
class Wav2Vec2ASRBundle(Wav2Vec2Bundle):
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - ASR
        >>> import torchaudio
        >>>
        >>> bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 1.18G/1.18G [00:17<00:00, 73.8MB/s]
        >>>
        >>> # Check the corresponding labels of the output.
        >>> labels = bundle.get_labels()
        >>> print(labels)
        ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Infer the label probability distribution
        >>> emissions, _ = model(waveform)
        >>>
        >>> # Pass emission to decoder
        >>> # `ctc_decode` is for illustration purpose only
        >>> transcripts = ctc_decode(emissions, labels)
    """  # noqa: E501

    _labels: Tuple[str, ...]
    _remove_aux_axis: Tuple[int, ...] = (1, 2, 3)

    def get_labels(
        self,
        *,
        blank: str = "-",
    ) -> Tuple[str, ...]:
        """The output class labels.

        The first is blank token, and it is customizable.

        Args:
            blank (str, optional): Blank token. (default: ``'-'``)

        Returns:
            Tuple[str, ...]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> from torchaudio.pipelines import HUBERT_ASR_LARGE as bundle
            >>> bundle.get_labels()
            ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        """  # noqa: E501
        return (blank, *self._labels)

    def _get_state_dict(self, dl_kwargs):
        return utils._get_state_dict(self._path, dl_kwargs, self._remove_aux_axis)


WAV2VEC2_BASE = Wav2Vec2Bundle(
    _path="wav2vec2_fairseq_base_ls960.pth",
    _params={
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_BASE.__doc__ = """Wav2vec 2.0 model ("base" architecture),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned.

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_BASE_10M = Wav2Vec2ASRBundle(
    _path="wav2vec2_fairseq_base_ls960_asr_ll10m.pth",
    _params={
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_BASE_10M.__doc__ = """Wav2vec 2.0 model ("base" architecture with an extra linear module),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 10 minutes of transcribed audio from *Libri-Light* dataset
:cite:`librilight` ("train-10min" subset).

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_BASE_100H = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_base_ls960_asr_ls100.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)

WAV2VEC2_ASR_BASE_100H.__doc__ = """Wav2vec 2.0 model ("base" architecture with an extra linear module),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 100 hours of transcribed audio from "train-clean-100" subset.

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_BASE_960H = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_BASE_960H.__doc__ = """Wav2vec 2.0 model ("base" architecture with an extra linear module),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on the same audio with the corresponding transcripts.

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_LARGE = Wav2Vec2Bundle(
    "wav2vec2_fairseq_large_ls960.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.2,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_LARGE.__doc__ = """Wav2vec 2.0 model ("large" architecture),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned.

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_10M = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_large_ls960_asr_ll10m.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.2,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_LARGE_10M.__doc__ = """Wav2vec 2.0 model ("large" architecture with an extra linear module),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 10 minutes of transcribed audio from *Libri-Light* dataset
:cite:`librilight` ("train-10min" subset).

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_100H = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_large_ls960_asr_ls100.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.2,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_LARGE_100H.__doc__ = """Wav2vec 2.0 model ("large" architecture with an extra linear module),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 100 hours of transcribed audio from
the same dataset ("train-clean-100" subset).

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_960H = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_large_ls960_asr_ls960.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.2,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_LARGE_960H.__doc__ = """Wav2vec 2.0 model ("large" architecture with an extra linear module),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on the same audio with the corresponding transcripts.

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa:  E501

WAV2VEC2_LARGE_LV60K = Wav2Vec2Bundle(
    "wav2vec2_fairseq_large_lv60k.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
WAV2VEC2_LARGE_LV60K.__doc__ = """Wav2vec 2.0 model ("large-lv60k" architecture),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* dataset :cite:`librilight`,
not fine-tuned.

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_LV60K_10M = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_large_lv60k_asr_ll10m.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_LARGE_LV60K_10M.__doc__ = """Wav2vec 2.0 model ("large-lv60k" architecture with an extra linear module),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* dataset :cite:`librilight`, and
fine-tuned for ASR on 10 minutes of transcribed audio from the same dataset ("train-10min" subset).

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_LV60K_100H = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_large_lv60k_asr_ls100.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_LARGE_LV60K_100H.__doc__ = """Wav2vec 2.0 model ("large-lv60k" architecture with an extra linear module),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* dataset :cite:`librilight`, and
fine-tuned for ASR on 100 hours of transcribed audio from
*LibriSpeech* dataset :cite:`7178964` ("train-clean-100" subset).

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_LV60K_960H = Wav2Vec2ASRBundle(
    "wav2vec2_fairseq_large_lv60k_asr_ls960.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
WAV2VEC2_ASR_LARGE_LV60K_960H.__doc__ = """Wav2vec 2.0 model ("large-lv60k" architecture with an extra linear module),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* :cite:`librilight` dataset, and
fine-tuned for ASR on 960 hours of transcribed audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").

Originally published by the authors of *wav2vec 2.0* :cite:`baevski2020wav2vec` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

WAV2VEC2_XLSR53 = Wav2Vec2Bundle(
    "wav2vec2_fairseq_large_xlsr53.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
WAV2VEC2_XLSR53.__doc__ = """Wav2vec 2.0 model ("base" architecture),
pre-trained on 56,000 hours of unlabeled audio from multiple datasets (
*Multilingual LibriSpeech* :cite:`Pratap_2020`,
*CommonVoice* :cite:`ardila2020common` and
*BABEL* :cite:`Gales2014SpeechRA`),
not fine-tuned.

Originally published by the authors of
*Unsupervised Cross-lingual Representation Learning for Speech Recognition*
:cite:`conneau2020unsupervised` under MIT License and redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

HUBERT_BASE = Wav2Vec2Bundle(
    "hubert_fairseq_base_ls960.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
HUBERT_BASE.__doc__ = """HuBERT model ("base" architecture),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned.

Originally published by the authors of *HuBERT* :cite:`hsu2021hubert` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

HUBERT_LARGE = Wav2Vec2Bundle(
    "hubert_fairseq_large_ll60k.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
HUBERT_LARGE.__doc__ = """HuBERT model ("large" architecture),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* dataset :cite:`librilight`,
not fine-tuned.

Originally published by the authors of *HuBERT* :cite:`hsu2021hubert` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

HUBERT_XLARGE = Wav2Vec2Bundle(
    "hubert_fairseq_xlarge_ll60k.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1280,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 48,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 5120,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
HUBERT_XLARGE.__doc__ = """HuBERT model ("extra large" architecture),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* dataset :cite:`librilight`,
not fine-tuned.

Originally published by the authors of *HuBERT* :cite:`hsu2021hubert` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501

HUBERT_ASR_LARGE = Wav2Vec2ASRBundle(
    "hubert_fairseq_large_ll60k_asr_ls960.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
HUBERT_ASR_LARGE.__doc__ = """HuBERT model ("large" architecture),
pre-trained on 60,000 hours of unlabeled audio from *Libri-Light* dataset :cite:`librilight`, and
fine-tuned for ASR on 960 hours of transcribed audio from *LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").

Originally published by the authors of *HuBERT* :cite:`hsu2021hubert` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

HUBERT_ASR_XLARGE = Wav2Vec2ASRBundle(
    "hubert_fairseq_xlarge_ll60k_asr_ls960.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1280,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 48,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 5120,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 29,
    },
    _labels=utils._get_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
HUBERT_ASR_XLARGE.__doc__ = """HuBERT model ("extra large" architecture),
pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset :cite:`librilight`, and
fine-tuned for ASR on 960 hours of transcribed audio from
*LibriSpeech* dataset :cite:`7178964`
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").

Originally published by the authors of *HuBERT* :cite:`hsu2021hubert` under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501


VOXPOPULI_ASR_BASE_10K_DE = Wav2Vec2ASRBundle(
    "wav2vec2_voxpopuli_base_10k_asr_de.pt",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 32,
    },
    _labels=utils._get_de_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _remove_aux_axis=(1, 2, 3, 35),
    _model_type="Wav2Vec2",
)
VOXPOPULI_ASR_BASE_10K_DE.__doc__ = """wav2vec 2.0 model ("base" architecture),
pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset :cite:`voxpopuli`
("10k" subset, consisting of 23 languages), and
fine-tuned for ASR on 282 hours of transcribed audio from "de" subset.

Originally published by the authors of *VoxPopuli* :cite:`voxpopuli` under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501


VOXPOPULI_ASR_BASE_10K_EN = Wav2Vec2ASRBundle(
    "wav2vec2_voxpopuli_base_10k_asr_en.pt",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 28,
    },
    _labels=utils._get_vp_en_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _remove_aux_axis=(1, 2, 3, 31),
    _model_type="Wav2Vec2",
)
VOXPOPULI_ASR_BASE_10K_EN.__doc__ = """wav2vec 2.0 model ("base" architecture),
pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset :cite:`voxpopuli`
("10k" subset, consisting of 23 languages), and
fine-tuned for ASR on 543 hours of transcribed audio from "en" subset.

Originally published by the authors of *VoxPopuli* :cite:`voxpopuli` under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501


VOXPOPULI_ASR_BASE_10K_ES = Wav2Vec2ASRBundle(
    "wav2vec2_voxpopuli_base_10k_asr_es.pt",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 35,
    },
    _labels=utils._get_es_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _remove_aux_axis=(1, 2, 3, 35),
    _model_type="Wav2Vec2",
)
VOXPOPULI_ASR_BASE_10K_ES.__doc__ = """wav2vec 2.0 model ("base" architecture),
pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset :cite:`voxpopuli`
("10k" subset, consisting of 23 languages), and
fine-tuned for ASR on 166 hours of transcribed audio from "es" subset.

Originally published by the authors of *VoxPopuli* :cite:`voxpopuli` under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501

VOXPOPULI_ASR_BASE_10K_FR = Wav2Vec2ASRBundle(
    "wav2vec2_voxpopuli_base_10k_asr_fr.pt",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 43,
    },
    _labels=utils._get_fr_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _model_type="Wav2Vec2",
)
VOXPOPULI_ASR_BASE_10K_FR.__doc__ = """wav2vec 2.0 model ("base" architecture),
pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset :cite:`voxpopuli`
("10k" subset, consisting of 23 languages), and
fine-tuned for ASR on 211 hours of transcribed audio from "fr" subset.

Originally published by the authors of *VoxPopuli* :cite:`voxpopuli` under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501


VOXPOPULI_ASR_BASE_10K_IT = Wav2Vec2ASRBundle(
    "wav2vec2_voxpopuli_base_10k_asr_it.pt",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 37,
    },
    _labels=utils._get_it_labels(),
    _sample_rate=16000,
    _normalize_waveform=False,
    _remove_aux_axis=(1, 2, 3),
    _model_type="Wav2Vec2",
)
VOXPOPULI_ASR_BASE_10K_IT.__doc__ = """wav2vec 2.0 model ("base" architecture),
pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset :cite:`voxpopuli`
("10k" subset, consisting of 23 languages), and
fine-tuned for ASR on 91 hours of transcribed audio from "it" subset.

Originally published by the authors of *VoxPopuli* :cite:`voxpopuli` under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501


WAVLM_BASE = Wav2Vec2Bundle(
    "wavlm_base.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_max_distance": 800,
        "encoder_num_buckets": 320,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": None,
    },
    _model_type="WavLM",
    _sample_rate=16000,
    _normalize_waveform=False,
)
WAVLM_BASE.__doc__ = """WavLM Base model ("base" architecture),
pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset :cite:`7178964`, not fine-tuned.

Originally published by the authors of *WavLM* :cite:`chen2022wavlm` under MIT License and
redistributed with the same license.
[`License <https://github.com/microsoft/unilm/blob/65f15af2a307ebb64cfb25adf54375b002e6fe8d/LICENSE>`__,
`Source <https://github.com/microsoft/unilm/tree/65f15af2a307ebb64cfb25adf54375b002e6fe8d/wavlm#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501


WAVLM_BASE_PLUS = Wav2Vec2Bundle(
    "wavlm_base_plus.pth",
    {
        "extractor_mode": "group_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_max_distance": 800,
        "encoder_num_buckets": 320,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.05,
        "aux_num_out": None,
    },
    _model_type="WavLM",
    _sample_rate=16000,
    _normalize_waveform=False,
)
WAVLM_BASE_PLUS.__doc__ = """WavLM Base+ model ("base" architecture),
pre-trained on 60,000 hours of Libri-Light dataset :cite:`librilight`, 10,000 hours of GigaSpeech :cite:`GigaSpeech2021`,
and 24,000 hours of *VoxPopuli* :cite:`voxpopuli`, not fine-tuned.

Originally published by the authors of *WavLM* :cite:`chen2022wavlm` under MIT License and
redistributed with the same license.
[`License <https://github.com/microsoft/unilm/blob/65f15af2a307ebb64cfb25adf54375b002e6fe8d/LICENSE>`__,
`Source <https://github.com/microsoft/unilm/tree/65f15af2a307ebb64cfb25adf54375b002e6fe8d/wavlm#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501


WAVLM_LARGE = Wav2Vec2Bundle(
    "wavlm_large.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": False,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_max_distance": 800,
        "encoder_num_buckets": 320,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.05,
        "aux_num_out": None,
    },
    _model_type="WavLM",
    _sample_rate=16000,
    _normalize_waveform=True,
)
WAVLM_LARGE.__doc__ = """WavLM Large model ("large" architecture),
pre-trained on 60,000 hours of Libri-Light dataset :cite:`librilight`, 10,000 hours of GigaSpeech :cite:`GigaSpeech2021`,
and 24,000 hours of *VoxPopuli* :cite:`voxpopuli`, not fine-tuned.

Originally published by the authors of *WavLM* :cite:`chen2022wavlm` under MIT License and
redistributed with the same license.
[`License <https://github.com/microsoft/unilm/blob/65f15af2a307ebb64cfb25adf54375b002e6fe8d/LICENSE>`__,
`Source <https://github.com/microsoft/unilm/tree/65f15af2a307ebb64cfb25adf54375b002e6fe8d/wavlm#pre-trained-models>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
"""  # noqa: E501


WAV2VEC2_XLSR_300M = Wav2Vec2Bundle(
    "wav2vec2_xlsr_300m.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _model_type="Wav2Vec2",
    _sample_rate=16000,
    _normalize_waveform=True,
)
WAV2VEC2_XLSR_300M.__doc__ = """XLS-R model with 300 million parameters,
pre-trained on 436,000 hours of unlabeled audio from multiple datasets (
*Multilingual LibriSpeech* :cite:`Pratap_2020`,
*CommonVoice* :cite:`ardila2020common`,
*VoxLingua107* :cite:`valk2021voxlingua107`,
*BABEL* :cite:`Gales2014SpeechRA`, and
*VoxPopuli* :cite:`voxpopuli`) in 128 languages,
not fine-tuned.

Originally published by the authors of *XLS-R* :cite:`babu2021xls` under MIT License and
redistributed with the same license.
[`License <https://github.com/facebookresearch/fairseq/blob/30c912b73c0f88d41171879b2f03226a171004ef/LICENSE>`__,
`Source <https://github.com/facebookresearch/fairseq/tree/30c912b73c0f88d41171879b2f03226a171004ef/examples/wav2vec/xlsr#xls-r>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for usage details.
"""  # noqa: E501


WAV2VEC2_XLSR_1B = Wav2Vec2Bundle(
    "wav2vec2_xlsr_1b.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1280,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 48,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 5120,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _model_type="Wav2Vec2",
    _sample_rate=16000,
    _normalize_waveform=True,
)
WAV2VEC2_XLSR_1B.__doc__ = """XLS-R model with 1 billion parameters,
pre-trained on 436,000 hours of unlabeled audio from multiple datasets (
*Multilingual LibriSpeech* :cite:`Pratap_2020`,
*CommonVoice* :cite:`ardila2020common`,
*VoxLingua107* :cite:`valk2021voxlingua107`,
*BABEL* :cite:`Gales2014SpeechRA`, and
*VoxPopuli* :cite:`voxpopuli`) in 128 languages,
not fine-tuned.

Originally published by the authors of *XLS-R* :cite:`babu2021xls` under MIT License and
redistributed with the same license.
[`License <https://github.com/facebookresearch/fairseq/blob/30c912b73c0f88d41171879b2f03226a171004ef/LICENSE>`__,
`Source <https://github.com/facebookresearch/fairseq/tree/30c912b73c0f88d41171879b2f03226a171004ef/examples/wav2vec/xlsr#xls-r>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for usage details.
"""  # noqa: E501

WAV2VEC2_XLSR_2B = Wav2Vec2Bundle(
    "wav2vec2_xlsr_2b.pth",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1920,
        "encoder_projection_dropout": 0.1,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 48,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 7680,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None,
    },
    _model_type="Wav2Vec2",
    _sample_rate=16000,
    _normalize_waveform=True,
)
WAV2VEC2_XLSR_2B.__doc__ = """XLS-R model with 2 billion parameters,
pre-trained on 436,000 hours of unlabeled audio from multiple datasets (
*Multilingual LibriSpeech* :cite:`Pratap_2020`,
*CommonVoice* :cite:`ardila2020common`,
*VoxLingua107* :cite:`valk2021voxlingua107`,
*BABEL* :cite:`Gales2014SpeechRA`, and
*VoxPopuli* :cite:`voxpopuli`) in 128 languages,
not fine-tuned.

Originally published by the authors of *XLS-R* :cite:`babu2021xls` under MIT License and
redistributed with the same license.
[`License <https://github.com/facebookresearch/fairseq/blob/30c912b73c0f88d41171879b2f03226a171004ef/LICENSE>`__,
`Source <https://github.com/facebookresearch/fairseq/tree/30c912b73c0f88d41171879b2f03226a171004ef/examples/wav2vec/xlsr#xls-r>`__]

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2Bundle` for usage details.
"""  # noqa: E501


@dataclass
class Wav2Vec2FABundle(Wav2Vec2ASRBundle):
    """Data class that bundles associated information to use pretrained :py:class:`~torchaudio.models.Wav2Vec2Model` for forced alignment.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - Feature Extraction
        >>> import torchaudio
        >>>
        >>> bundle = torchaudio.pipelines.MMS_FA
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 1.18G/1.18G [00:05<00:00, 216MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Estimate the probability of token distribution
        >>> emission, _ = model(waveform)
        >>>
        >>> # Generate frame-wise alignment
        >>> alignment, scores = torchaudio.functional.forced_align(
        >>>     emission, targets, input_lengths, target_lengths, blank=0)
        >>>
    """  # noqa: E501

    class Tokenizer(aligner.ITokenizer):
        """Interface of the tokenizer"""

    class Aligner(aligner.IAligner):
        """Interface of the aligner"""

    def get_labels(self, star: Optional[str] = "*", blank: str = "-") -> Tuple[str, ...]:
        """Get the labels corresponding to the feature dimension of emission.

        The first is blank token, and it is customizable.

        Args:
            star (str or None, optional): Change or disable star token. (default: ``"*"``)
            blank (str, optional): Change the blank token. (default: ``'-'``)

        Returns:
            Tuple[str, ...]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> from torchaudio.pipelines import MMS_FA as bundle
            >>> bundle.get_labels()
            ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x', '*')
            >>> bundle.get_labels(star=None)
            ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x')
        """  # noqa: E501
        labels = super().get_labels(blank=blank)
        return labels if star is None else (*labels, star)

    def get_model(self, with_star: bool = True, *, dl_kwargs=None) -> Module:
        """Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            with_star (bool, optional): If enabled, the last dimension of output layer is
                extended by one, which corresponds to `star` token.
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.models.Wav2Vec2Model`.

            .. note::

               The model created with this method returns probability in log-domain,
               (i.e. :py:func:`torch.nn.functional.log_softmax` is applied), whereas
               the other Wav2Vec2 models returns logit.
        """
        model = utils._get_model(self._model_type, self._params)
        state_dict = utils._get_state_dict(self._path, dl_kwargs, self._remove_aux_axis)
        model.load_state_dict(state_dict)
        model = utils._extend_model(
            model, normalize_waveform=self._normalize_waveform, apply_log_softmax=True, append_star=with_star
        )
        model.eval()
        return model

    def get_dict(self, star: Optional[str] = "*", blank: str = "-") -> Dict[str, int]:
        """Get the mapping from token to index (in emission feature dim)

        Args:
            star (str or None, optional): Change or disable star token. (default: ``"*"``)
            blank (str, optional): Change the blank token. (default: ``'-'``)

        Returns:
            Tuple[str, ...]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> from torchaudio.pipelines import MMS_FA as bundle
            >>> bundle.get_dict()
            {'-': 0, 'a': 1, 'i': 2, 'e': 3, 'n': 4, 'o': 5, 'u': 6, 't': 7, 's': 8, 'r': 9, 'm': 10, 'k': 11, 'l': 12, 'd': 13, 'g': 14, 'h': 15, 'y': 16, 'b': 17, 'p': 18, 'w': 19, 'c': 20, 'v': 21, 'j': 22, 'z': 23, 'f': 24, "'": 25, 'q': 26, 'x': 27, '*': 28}
            >>> bundle.get_dict(star=None)
            {'-': 0, 'a': 1, 'i': 2, 'e': 3, 'n': 4, 'o': 5, 'u': 6, 't': 7, 's': 8, 'r': 9, 'm': 10, 'k': 11, 'l': 12, 'd': 13, 'g': 14, 'h': 15, 'y': 16, 'b': 17, 'p': 18, 'w': 19, 'c': 20, 'v': 21, 'j': 22, 'z': 23, 'f': 24, "'": 25, 'q': 26, 'x': 27}
        """  # noqa: E501
        return {k: i for i, k in enumerate(self.get_labels(star=star, blank=blank))}

    def get_tokenizer(self) -> Tokenizer:
        """Instantiate a Tokenizer.

        Returns:
            Tokenizer
        """
        return aligner.Tokenizer(self.get_dict())

    def get_aligner(self) -> Aligner:
        """Instantiate an Aligner.

        Returns:
            Aligner
        """
        return aligner.Aligner(blank=0)


MMS_FA = Wav2Vec2FABundle(
    "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
    {
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.1,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.1,
        "aux_num_out": 28,
    },
    _labels=utils._get_mms_labels(),
    _sample_rate=16000,
    _normalize_waveform=True,
    _model_type="Wav2Vec2",
)
MMS_FA.__doc__ = """
Trained on 31K hours of data in 1,130 languages from *Scaling Speech Technology to 1,000+ Languages* :cite:`pratap2023scaling`.

Published by the authors of *Scaling Speech Technology to 1,000+ Languages* :cite:`pratap2023scaling` under [`CC-BY-NC 4.0 License <https://github.com/facebookresearch/fairseq/tree/100cd91db19bb27277a06a25eb4154c805b10189/examples/mms#license>`__].

Please refer to :py:class:`torchaudio.pipelines.Wav2Vec2FABundle` for usage details.

.. note::

   Unlike other Wav2Vec2 bundles, this model does not have a token for word boundary (like `|`). This makes the post-processing of alignments slightly different.
"""  # noqa: E501
