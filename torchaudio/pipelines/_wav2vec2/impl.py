from dataclasses import dataclass
from typing import Dict, Tuple, Any

import torch
from torchaudio._internal import load_state_dict_from_url
from torchaudio.models import wav2vec2_model, Wav2Vec2Model

from . import utils


__all__ = []


@dataclass
class Wav2Vec2Bundle:
    """torchaudio.pipelines.Wav2Vec2Bundle()

    Data class that bundles associated information to use pretrained Wav2Vec2Model.

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

    @property
    def sample_rate(self) -> float:
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate

    def _get_state_dict(self, dl_kwargs):
        url = f"https://download.pytorch.org/torchaudio/models/{self._path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict

    def get_model(self, *, dl_kwargs=None) -> Wav2Vec2Model:
        # Overriding the signature so that the return type is correct on Sphinx
        """get_model(self, *, dl_kwargs=None) -> torchaudio.models.Wav2Vec2Model

        Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.
        """
        model = wav2vec2_model(**self._params)
        model.load_state_dict(self._get_state_dict(dl_kwargs))
        model.eval()
        return model


@dataclass
class Wav2Vec2ASRBundle(Wav2Vec2Bundle):
    """torchaudio.pipelines.Wav2Vec2ASRBundle()

    Data class that bundles associated information to use pretrained Wav2Vec2Model.

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

    _labels: Tuple[str]
    _remove_aux_axis: Tuple[int] = (1, 2, 3)

    def get_labels(
        self,
        *,
        blank: str = "-",
    ) -> Tuple[str]:
        """The output class labels (only applicable to fine-tuned bundles)

        The first is blank token, and it is customizable.

        Args:
            blank (str, optional): Blank token. (default: ``'-'``)

        Returns:
            Tuple[str]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> import torchaudio
            >>> torchaudio.models.HUBERT_ASR_LARGE.get_labels()
            ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        """  # noqa: E501
        return (blank, *self._labels)

    def _get_state_dict(self, dl_kwargs):
        state_dict = super()._get_state_dict(dl_kwargs)
        if self._remove_aux_axis:
            # Remove the seemingly unnecessary axis
            # For ASR task, the pretrained weights originated from fairseq has unrelated dimensions at index 1, 2, 3
            # It's originated from the Dictionary implementation of fairseq, which was intended for NLP tasks,
            # but not used during the ASR training.
            # https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/data/dictionary.py#L21-L37
            # https://github.com/pytorch/fairseq/blob/c5ff181125c7e6126b49a85e5ebdd5f5b6a07914/fairseq/criterions/ctc.py#L126-L129
            #
            # Also, some pretrained weights originated from voxpopuli has an extra dimensions that almost never used and
            # that resembles mistake.
            # The label `1` shows up in the training dataset of German (1 out of 16M),
            # English (1 / 28M), Spanish (1 / 9.4M), Romanian (1 / 4.7M) and Polish (6 / 5.8M)
            for key in ["aux.weight", "aux.bias"]:
                t = state_dict[key]
                state_dict[key] = torch.stack([t[i] for i in range(t.size(0)) if i not in self._remove_aux_axis])
        return state_dict


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
)
WAV2VEC2_BASE.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").
Not fine-tuned.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
WAV2VEC2_ASR_BASE_10M.__doc__ = """Build "base" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 10 minutes of transcribed audio from *Libri-Light* dataset
[:footcite:`librilight`] ("train-10min" subset).

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)

WAV2VEC2_ASR_BASE_100H.__doc__ = """Build "base" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 100 hours of transcribed audio from "train-clean-100" subset.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_ASR_BASE_960H.__doc__ = """Build "base" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on the same audio with the corresponding transcripts.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_LARGE.__doc__ = """Build "large" wav2vec2 model.

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").
Not fine-tuned.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
WAV2VEC2_ASR_LARGE_10M.__doc__ = """Build "large" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 10 minutes of transcribed audio from *Libri-Light* dataset
[:footcite:`librilight`] ("train-10min" subset).

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_ASR_LARGE_100H.__doc__ = """Build "large" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 100 hours of transcribed audio from
the same dataset ("train-clean-100" subset).

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_ASR_LARGE_960H.__doc__ = """Build "large" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on the same audio with the corresponding transcripts.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_LARGE_LV60K.__doc__ = """Build "large-lv60k" wav2vec2 model.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`].
Not fine-tuned.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
WAV2VEC2_ASR_LARGE_LV60K_10M.__doc__ = """Build "large-lv60k" wav2vec2 model with an extra linear module

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`], and
fine-tuned for ASR on 10 minutes of transcribed audio from
the same dataset ("train-10min" subset).

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_ASR_LARGE_LV60K_100H.__doc__ = """Build "large-lv60k" wav2vec2 model with an extra linear module

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`], and
fine-tuned for ASR on 100 hours of transcribed audio from
*LibriSpeech* dataset [:footcite:`7178964`] ("train-clean-100" subset).

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_ASR_LARGE_LV60K_960H.__doc__ = """Build "large-lv60k" wav2vec2 model with an extra linear module

Pre-trained on 60,000 hours of unlabeled audio from *Libri-Light*
[:footcite:`librilight`] dataset, and
fine-tuned for ASR on 960 hours of transcribed audio from
*LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
WAV2VEC2_XLSR53.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Trained on 56,000 hours of unlabeled audio from multiple datasets (
*Multilingual LibriSpeech* [:footcite:`Pratap_2020`],
*CommonVoice* [:footcite:`ardila2020common`] and
*BABEL* [:footcite:`Gales2014SpeechRA`]).
Not fine-tuned.

Originally published by the authors of
*Unsupervised Cross-lingual Representation Learning for Speech Recognition*
[:footcite:`conneau2020unsupervised`] under MIT License and redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
HUBERT_BASE.__doc__ = """HuBERT model with "Base" configuration.

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").
Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
HUBERT_LARGE.__doc__ = """HuBERT model with "Large" configuration.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`].
Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
HUBERT_XLARGE.__doc__ = """HuBERT model with "Extra Large" configuration.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`].
Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2Bundle` for the usage.
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
)
HUBERT_ASR_LARGE.__doc__ = """HuBERT model with "Large" configuration.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`], and
fine-tuned for ASR on 960 hours of transcribed audio from
*LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
HUBERT_ASR_XLARGE.__doc__ = """HuBERT model with "Extra Large" configuration.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`], and
fine-tuned for ASR on 960 hours of transcribed audio from
*LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
    _remove_aux_axis=(1, 2, 3, 35),
)
VOXPOPULI_ASR_BASE_10K_DE.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset [:footcite:`voxpopuli`]
("10k" subset, consisting of 23 languages).
Fine-tuned for ASR on 282 hours of transcribed audio from "de" subset.

Originally published by the authors of *VoxPopuli* [:footcite:`voxpopuli`] under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
    _remove_aux_axis=(1, 2, 3, 31),
)
VOXPOPULI_ASR_BASE_10K_EN.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset [:footcite:`voxpopuli`]
("10k" subset, consisting of 23 languages).

Fine-tuned for ASR on 543 hours of transcribed audio from "en" subset.
Originally published by the authors of *VoxPopuli* [:footcite:`voxpopuli`] under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
    _remove_aux_axis=(1, 2, 3, 35),
)
VOXPOPULI_ASR_BASE_10K_ES.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset [:footcite:`voxpopuli`]
("10k" subset, consisting of 23 languages).
Fine-tuned for ASR on 166 hours of transcribed audio from "es" subset.

Originally published by the authors of *VoxPopuli* [:footcite:`voxpopuli`] under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
)
VOXPOPULI_ASR_BASE_10K_FR.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset [:footcite:`voxpopuli`]
("10k" subset, consisting of 23 languages).
Fine-tuned for ASR on 211 hours of transcribed audio from "fr" subset.

Originally published by the authors of *VoxPopuli* [:footcite:`voxpopuli`] under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
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
    _remove_aux_axis=(1, 2, 3),
)
VOXPOPULI_ASR_BASE_10K_IT.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 10k hours of unlabeled audio from *VoxPopuli* dataset [:footcite:`voxpopuli`]
("10k" subset, consisting of 23 languages).
Fine-tuned for ASR on 91 hours of transcribed audio from "it" subset.

Originally published by the authors of *VoxPopuli* [:footcite:`voxpopuli`] under CC BY-NC 4.0 and
redistributed with the same license.
[`License <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#license>`__,
`Source <https://github.com/facebookresearch/voxpopuli/tree/160e4d7915bad9f99b2c35b1d3833e51fd30abf2#asr-and-lm>`__]

Please refer to :func:`torchaudio.pipelines.Wav2Vec2ASRBundle` for the usage.
"""  # noqa: E501
