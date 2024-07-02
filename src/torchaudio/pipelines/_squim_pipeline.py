from dataclasses import dataclass

import torch
import torchaudio

from torchaudio.models import squim_objective_base, squim_subjective_base, SquimObjective, SquimSubjective


@dataclass
class SquimObjectiveBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.models.SquimObjective` model.

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
        >>> from torchaudio.pipelines import SQUIM_OBJECTIVE as bundle
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

    def get_model(self) -> SquimObjective:
        """Construct the SquimObjective model, and load the pretrained weight.

        Returns:
            Variation of :py:class:`~torchaudio.models.SquimObjective`.
        """
        model = squim_objective_base()
        path = torchaudio.utils.download_asset(f"models/{self._path}")
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict)
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
SQUIM_OBJECTIVE.__doc__ = """SquimObjective pipeline trained using approach described in
    :cite:`kumar2023torchaudio` on the *DNS 2020 Dataset* :cite:`reddy2020interspeech`.

    The underlying model is constructed by :py:func:`torchaudio.models.squim_objective_base`.
    The weights are under `Creative Commons Attribution 4.0 International License
    <https://github.com/microsoft/DNS-Challenge/blob/interspeech2020/master/LICENSE>`__.

    Please refer to :py:class:`SquimObjectiveBundle` for usage instructions.
    """


@dataclass
class SquimSubjectiveBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.models.SquimSubjective` model.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    This bundle can estimate subjective metric scores for speech enhancement, such as MOS.
    A typical use case would be a flow like `waveform -> score`. Please see below for the code example.

    Example: Estimate the subjective metric scores for the input waveform.
        >>> import torch
        >>> import torchaudio
        >>> from torchaudio.pipelines import SQUIM_SUBJECTIVE as bundle
        >>>
        >>> # Load the SquimSubjective bundle
        >>> model = bundle.get_model()
        Downloading: "https://download.pytorch.org/torchaudio/models/squim_subjective_bvcc_daps.pth"
        100%|████████████| 360M/360M [00:09<00:00, 41.1MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>> # Use a clean reference (doesn't need to be the reference for the waveform) as the second input
        >>> reference = torchaudio.functional.resample(reference, sample_rate, bundle.sample_rate)
        >>>
        >>> # Estimate subjective metric scores
        >>> score = model(waveform, reference)
        >>> print(f"MOS: {score}.")
    """  # noqa: E501

    _path: str
    _sample_rate: float

    def get_model(self) -> SquimSubjective:
        """Construct the SquimSubjective model, and load the pretrained weight.
        Returns:
            Variation of :py:class:`~torchaudio.models.SquimObjective`.
        """
        model = squim_subjective_base()
        path = torchaudio.utils.download_asset(f"models/{self._path}")
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @property
    def sample_rate(self):
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate


SQUIM_SUBJECTIVE = SquimSubjectiveBundle(
    "squim_subjective_bvcc_daps.pth",
    _sample_rate=16000,
)
SQUIM_SUBJECTIVE.__doc__ = """SquimSubjective pipeline trained
    as described in :cite:`manocha2022speech` and :cite:`kumar2023torchaudio`
    on the *BVCC* :cite:`cooper2021voices` and *DAPS* :cite:`mysore2014can` datasets.

    The underlying model is constructed by :py:func:`torchaudio.models.squim_subjective_base`.
    The weights are under `Creative Commons Attribution Non Commercial 4.0 International
    <https://zenodo.org/record/4660670#.ZBtWPOxuerN>`__.

    Please refer to :py:class:`SquimSubjectiveBundle` for usage instructions.
    """
