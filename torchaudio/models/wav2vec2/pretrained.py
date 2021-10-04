from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

from torch.hub import load_state_dict_from_url

from .model import _get_model, Wav2Vec2Model

__all__ = []


@dataclass
class Wav2Vec2PretrainedModelBundle:
    """torchaudio.models.Wav2Vec2PretrainedModelBundle()

    Data class that bundles associated information to use pretrained Wav2Vec2Model.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    different pretrained models. Client code should access pretrained models via thsse
    instances.

    Please see bellow for the usage and the availale values.

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

    Example - Model finu-tuned for ASR
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
            **dl_kwargs: Passed to :func:`torch.hub.load_state_dict_from_url`.
        """
        model = _get_model(**self._params)
        url = f'https://download.pytorch.org/models/audio/{self._path}'
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        model.load_state_dict(state_dict)
        return model

    @property
    def labels(self):
        """For ASR bundles, this represents the label of the output classes."""
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

Trained on 960 hours of *LibriSpeech* [:footcite:`7178964`] dataset. Not fine-tuned.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`].
[`Source <https://github.com/pytorch/fairseq/tree/main/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]
"""

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

Pre-trained on 60,000 hours of *Libri-Light* [:footcite:`librilight`] dataset, and
fine-tuned for ASR on 960 hours of *LibriSpeech* [:footcite:`7178964`] dataset.

Originally published by the authors of *HuBERT* [:footcite:`hsu2021hubert`].
[`Source <https://github.com/pytorch/fairseq/tree/main/examples/hubert#pre-trained-and-fine-tuned-asr-models>`__]
"""
