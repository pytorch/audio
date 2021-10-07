from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

from torch.hub import load_state_dict_from_url

from .model import wav2vec2_model, Wav2Vec2Model

__all__ = []


@dataclass
class Wav2Vec2PretrainedModelBundle:
    """torchaudio.models.Wav2Vec2PretrainedModelBundle()

    Data class that bundles associated information to use pretrained Wav2Vec2Model.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - Pretraining model
        >>> import torchaudio
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = torchaudio.models.HUBERT_BASE.get_model()
        Downloading:
        100%|███████████████████████████████| 360M/360M [00:06<00:00, 60.6MB/s]
        >>> # Extract acoustic features
        >>> waveform, sample_rate = torchaudio.load('my_speech.mp3')
        >>> features, _ = model.extract_features(waveform)

    Example - Model fine-tuned for ASR
        >>> import torchaudio
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = torchaudio.models.HUBERT_ASR_LARGE.get_model()
        Downloading:
        100%|███████████████████████████████| 1.18G/1.18G [00:17<00:00, 73.8MB/s]
        >>> # Check the corresponding labels of the output.
        >>> labels = torchaudio.models.HUBERT_ASR_LARGE.labels
        >>> print(labels)
        ('<s>', '<pad>', '</s>', '<unk>', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')
        >>> # Infer the label probability distribution
        >>> waveform, sample_rate = torchaudio.load('my_speech.mp3')
        >>> emissions, _ = model(waveform)
        >>> # Pass emission to decoder
        >>> # `ctc_decode` is for illustration purpose only
        >>> transcripts = ctc_decode(emissions, labels)

    """  # noqa: E501
    _path: str
    _params: Dict[str, Any]
    _labels: Optional[Tuple[str]]

    def get_model(self, *, dl_kwargs=None) -> Wav2Vec2Model:
        """Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.
        """
        model = wav2vec2_model(**self._params)
        url = f'https://download.pytorch.org/models/audio/{self._path}'
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        model.load_state_dict(state_dict)
        return model

    @property
    def labels(self) -> Optional[Tuple[str]]:
        """The optional output class labels (only applicable to ASR bundles)

        Returns:
            Tuple of strings or None:
            For fine-tuned ASR models, returns the tuple of strings representing
            the output class labels. For non-ASR models, the value is ``None``.
        """
        return self._labels


def _get_labels():
    return (
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '|',
        'E',
        'T',
        'A',
        'O',
        'N',
        'I',
        'H',
        'S',
        'R',
        'D',
        'L',
        'U',
        'M',
        'W',
        'C',
        'F',
        'G',
        'Y',
        'P',
        'B',
        'V',
        'K',
        "'",
        'X',
        'J',
        'Q',
        'Z',
    )


WAV2VEC2_BASE = Wav2Vec2PretrainedModelBundle(
    _path='wav2vec2_fairseq_base_ls960.pth',
    _params={
        'extractor_mode': 'group_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 768,
        'encoder_projection_dropout': 0.1,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 12,
        'encoder_num_heads': 12,
        'encoder_attention_dropout': 0.1,
        'encoder_ff_interm_features': 3072,
        'encoder_ff_interm_dropout': 0.0,
        'encoder_dropout': 0.1,
        'encoder_layer_norm_first': False,
        'encoder_layer_drop': 0.05,
        "aux_num_out": None,
    },
    _labels=None,
)
WAV2VEC2_BASE.__doc__ = """wav2vec 2.0 model with "Base" configuration.

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").
Not fine-tuned.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
"""  # noqa: E501

WAV2VEC2_ASR_BASE_10M = Wav2Vec2PretrainedModelBundle(
    _path='wav2vec2_fairseq_base_ls960_asr_ll10m.pth',
    _params={
        'extractor_mode': 'group_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 768,
        'encoder_projection_dropout': 0.1,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 12,
        'encoder_num_heads': 12,
        'encoder_attention_dropout': 0.1,
        'encoder_ff_interm_features': 3072,
        'encoder_ff_interm_dropout': 0.0,
        'encoder_dropout': 0.1,
        'encoder_layer_norm_first': False,
        'encoder_layer_drop': 0.05,
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

WAV2VEC2_ASR_BASE_100H = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_base_ls960_asr_ls100.pth',
    {
        'extractor_mode': 'group_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 768,
        'encoder_projection_dropout': 0.1,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 12,
        'encoder_num_heads': 12,
        'encoder_attention_dropout': 0.1,
        'encoder_ff_interm_features': 3072,
        'encoder_ff_interm_dropout': 0.0,
        'encoder_dropout': 0.1,
        'encoder_layer_norm_first': False,
        'encoder_layer_drop': 0.05,
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
)

WAV2VEC2_ASR_BASE_100H.__doc__ = """Build "base" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on 100 hours of transcribed audio from "train-clean-100" subset.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
"""  # noqa: E501

WAV2VEC2_ASR_BASE_960H = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_base_ls960_asr_ls960.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
)
WAV2VEC2_ASR_BASE_960H.__doc__ = """Build "base" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on the same audio with the corresponding transcripts.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
"""  # noqa: E501

WAV2VEC2_LARGE = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_ls960.pth',
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
    _labels=None,
)
WAV2VEC2_LARGE.__doc__ = """Build "large" wav2vec2 model.

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").
Not fine-tuned.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_10M = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_ls960_asr_ll10m.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_100H = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_ls960_asr_ls100.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_960H = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_ls960_asr_ls960.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
)
WAV2VEC2_ASR_LARGE_960H.__doc__ = """Build "large" wav2vec2 model with an extra linear module

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500"), and
fine-tuned for ASR on the same audio with the corresponding transcripts.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
"""  # noqa:  E501

WAV2VEC2_LARGE_LV60K = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_lv60k.pth',
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
    _labels=None,
)
WAV2VEC2_LARGE_LV60K.__doc__ = """Build "large-lv60k" wav2vec2 model.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`].
Not fine-tuned.

Originally published by the authors of *wav2vec 2.0* [:footcite:`baevski2020wav2vec`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/wav2vec#pre-trained-models>`__]
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_LV60K_10M = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_lv60k_asr_ll10m.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_LV60K_100H = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_lv60k_asr_ls100.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

WAV2VEC2_ASR_LARGE_LV60K_960H = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_lv60k_asr_ls960.pth',
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
        "aux_num_out": 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

WAV2VEC2_XLSR53 = Wav2Vec2PretrainedModelBundle(
    'wav2vec2_fairseq_large_xlsr53.pth',
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
    _labels=None,
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
"""  # noqa: E501

HUBERT_BASE = Wav2Vec2PretrainedModelBundle(
    'hubert_fairseq_base_ls960.pth',
    {
        'extractor_mode': 'group_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 768,
        'encoder_projection_dropout': 0.1,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 12,
        'encoder_num_heads': 12,
        'encoder_attention_dropout': 0.1,
        'encoder_ff_interm_features': 3072,
        'encoder_ff_interm_dropout': 0.0,
        'encoder_dropout': 0.1,
        'encoder_layer_norm_first': False,
        'encoder_layer_drop': 0.05,
        'aux_num_out': None,
    },
    _labels=None,
)
HUBERT_BASE.__doc__ = """HuBERT model with "Base" configuration.

Pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset [:footcite:`7178964`]
(the combination of "train-clean-100", "train-clean-360", and "train-other-500").
Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]
"""  # noqa: E501

HUBERT_LARGE = Wav2Vec2PretrainedModelBundle(
    'hubert_fairseq_large_ll60k.pth',
    {
        'extractor_mode': 'layer_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 1024,
        'encoder_projection_dropout': 0.0,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 24,
        'encoder_num_heads': 16,
        'encoder_attention_dropout': 0.0,
        'encoder_ff_interm_features': 4096,
        'encoder_ff_interm_dropout': 0.0,
        'encoder_dropout': 0.0,
        'encoder_layer_norm_first': True,
        'encoder_layer_drop': 0.0,
        'aux_num_out': None,
    },
    _labels=None,
)
HUBERT_LARGE.__doc__ = """HuBERT model with "Large" configuration.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`].
Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]
"""  # noqa: E501

HUBERT_XLARGE = Wav2Vec2PretrainedModelBundle(
    'hubert_fairseq_xlarge_ll60k.pth',
    {
        'extractor_mode': 'layer_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 1280,
        'encoder_projection_dropout': 0.0,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 48,
        'encoder_num_heads': 16,
        'encoder_attention_dropout': 0.0,
        'encoder_ff_interm_features': 5120,
        'encoder_ff_interm_dropout': 0.0,
        'encoder_dropout': 0.0,
        'encoder_layer_norm_first': True,
        'encoder_layer_drop': 0.0,
        'aux_num_out': None,
    },
    _labels=None,
)
HUBERT_XLARGE.__doc__ = """HuBERT model with "Extra Large" configuration.

Pre-trained on 60,000 hours of unlabeled audio from
*Libri-Light* dataset [:footcite:`librilight`].
Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`] under MIT License and
redistributed with the same license.
[`License <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/LICENSE>`__,
`Source <https://github.com/pytorch/fairseq/blob/ce6c9eeae163ac04b79539c78e74f292f29eaa18/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]
"""  # noqa: E501

HUBERT_ASR_LARGE = Wav2Vec2PretrainedModelBundle(
    'hubert_fairseq_large_ll60k_asr_ls960.pth',
    {
        'extractor_mode': 'layer_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 1024,
        'encoder_projection_dropout': 0.0,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 24,
        'encoder_num_heads': 16,
        'encoder_attention_dropout': 0.0,
        'encoder_ff_interm_features': 4096,
        'encoder_ff_interm_dropout': 0.1,
        'encoder_dropout': 0.0,
        'encoder_layer_norm_first': True,
        'encoder_layer_drop': 0.1,
        'aux_num_out': 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501

HUBERT_ASR_XLARGE = Wav2Vec2PretrainedModelBundle(
    'hubert_fairseq_xlarge_ll60k_asr_ls960.pth',
    {
        'extractor_mode': 'layer_norm',
        'extractor_conv_layer_config': [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        'extractor_conv_bias': False,
        'encoder_embed_dim': 1280,
        'encoder_projection_dropout': 0.0,
        'encoder_pos_conv_kernel': 128,
        'encoder_pos_conv_groups': 16,
        'encoder_num_layers': 48,
        'encoder_num_heads': 16,
        'encoder_attention_dropout': 0.0,
        'encoder_ff_interm_features': 5120,
        'encoder_ff_interm_dropout': 0.1,
        'encoder_dropout': 0.0,
        'encoder_layer_norm_first': True,
        'encoder_layer_drop': 0.1,
        'aux_num_out': 32,
    },
    _labels=_get_labels(),
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
"""  # noqa: E501
