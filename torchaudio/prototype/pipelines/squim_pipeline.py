from dataclasses import dataclass

from torchaudio._internal import load_state_dict_from_url

from torchaudio.prototype.models import squim_objective_base, SquimObjective


@dataclass
class SquimObjectiveBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.prototype.models.SquimObjective` model.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    This bundle can estimate objective metric scores for speech enhancement, such as STOI, PESQ, Si-SDR.
    A typical use case would be a flow like `waveform -> list of scores`. Please see below for the code example.

    Example: Estimate the objective metric scores for the input waveform.
        >>> import torch
        >>> import torchaudio
        >>> # Since SquimObjective bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE as bundle
        >>>
        >>> # Load the SquimObjective bundle
        >>> model = bundle.get_model()
        Downloading: "https://download.pytorch.org/torchaudio/models/squim_objective_dns2020.pth"
        100%|████████████| 28.2M/28.2M [00:03<00:00, 9.24MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Estimate objective metric scores
        >>> scores = model(waveform)
        >>> print(f"STOI: {scores[0].item()},  PESQ: {scores[1].item()}, SI-SDR: {scores[2].item()}.")
    """  # noqa: E501

    _path: str
    _sample_rate: float

    def _get_state_dict(self, dl_kwargs):
        url = f"https://download.pytorch.org/torchaudio/models/{self._path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict

    def get_model(self, *, dl_kwargs=None) -> SquimObjective:
        """Construct the SquimObjective model, and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.prototype.models.SquimObjective`.
        """
        model = squim_objective_base()
        model.load_state_dict(self._get_state_dict(dl_kwargs))
        model.eval()
        return model

    @property
    def sample_rate(self):
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate


SQUIM_OBJECTIVE = SquimObjectiveBundle(
    "squim_objective_dns2020.pth",
    _sample_rate=16000,
)
SQUIM_OBJECTIVE.__doc__ = """SquimObjective pipeline, trained on the *DNS 2020 Dataset*
    :cite:`reddy2020interspeech`.

    The underlying model is constructed by :py:func:`torchaudio.prototype.models.squim_objective_base`.
    The weights are under `Creative Commons Attribution 4.0 International License
    <https://github.com/microsoft/DNS-Challenge/blob/interspeech2020/master/LICENSE>`__.

    Please refer to :py:class:`SquimObjectiveBundle` for usage instructions.
    """
