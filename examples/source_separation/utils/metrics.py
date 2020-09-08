import math
from typing import Tuple
from itertools import permutations

import torch


def sdr(estimate: torch.Tensor, reference: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
    """Computes source-to-distortion ratio.

    1. scale the reference signal with power(s_est * s_ref) / powr(s_ref * s_ref)
    2. compute SNR between adjusted estimate and reference.

    Args:
        estimate (torch.Tensor): Estimtaed signal.
            Shape: [batch, speakers (can be 1), time frame]
        reference (torch.Tensor): Reference signal.
            Shape: [batch, speakers, time frame]
        epsilon (float): constant value used to stabilize division.

    Returns:
        torch.Tensor: scale-invariant source-to-distortion ratio.
        Shape: [batch, speaker]

    References:
        - Single-channel multi-speaker separation using deep clustering
          Y. Isik, J. Le Roux, Z. Chen, S. Watanabe, and J. R. Hershey,
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454

    Notes:
        This function is tested to produce the exact same result as
        https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py#L34-L56
    """
    reference_pow = reference.pow(2).mean(axis=2, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=2, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2).mean(axis=2)
    error_pow = error.pow(2).mean(axis=2)

    return 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)


class PIT(torch.nn.Module):
    """Applies utterance-level speaker permutation

    Computes the maxium possible value of the given utility function
    over the permutations of the speakers.

    Args:
        utility_func (function):
            Function that computes the utility (opposite of loss) with signature of
            (extimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor
            where input Tensors are shape of [batch, speakers, frame] and
            the output Tensor is shape of [batch, speakers].

    References:
        - Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of
          Deep Recurrent Neural Networks
          Morten Kolbæk, Dong Yu, Zheng-Hua Tan and Jesper Jensen
          https://arxiv.org/abs/1703.06284
    """

    def __init__(self, utility_func):
        super().__init__()
        self.utility_func = utility_func

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Compute utterance-level PIT Loss

        Args:
            estimate (torch.Tensor): Estimated source signals.
                Shape: [bacth, speakers, time frame]
            reference (torch.Tensor): Reference (original) source signals.
                Shape: [batch, speakers, time frame]

        Returns:
            torch.Tensor: Maximum criterion over the speaker permutation.
                Shape: [batch, ]
        """
        assert estimate.shape == reference.shape

        batch_size, num_speakers = reference.shape[:2]
        num_permute = math.factorial(num_speakers)

        util_mat = torch.zeros(
            batch_size, num_permute, dtype=estimate.dtype, device=estimate.device
        )
        for i, idx in enumerate(permutations(range(num_speakers))):
            util = self.utility_func(estimate, reference[:, idx, :])
            util_mat[:, i] = util.mean(dim=1)  # take the average over speaker dimension
        return util_mat.max(dim=1).values


_sdr_pit = PIT(utility_func=sdr)


def sdr_pit(estimate, reference):
    """Computes scale-invariant source-to-distortion ratio.

    1. adjust both estimate and reference to have 0-mean
    2. scale the reference signal with power(s_est * s_ref) / powr(s_ref * s_ref)
    3. compute SNR between adjusted estimate and reference.

    Args:
        estimate (torch.Tensor): Estimtaed signal.
            Shape: [batch, speakers (can be 1), time frame]
        reference (torch.Tensor): Reference signal.
            Shape: [batch, speakers, time frame]
        epsilon (float): constant value used to stabilize division.

    Returns:
        torch.Tensor: scale-invariant source-to-distortion ratio.
        Shape: [batch, speaker]

    References:
        - Single-channel multi-speaker separation using deep clustering
          Y. Isik, J. Le Roux, Z. Chen, S. Watanabe, and J. R. Hershey,
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454

    Notes:
        This function is tested to produce the exact same result as the reference implementation,
        *when the inputs have 0-mean*
        https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py#L107-L153
    """
    return _sdr_pit(estimate, reference)


def si_sdr_improvement(
    estimate: torch.Tensor, reference: torch.Tensor, mix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the improvement of scale-invariant SDR. (SI-SNRi)

    This function compute how much SDR is improved if the estimation is changed from
    the original mixture signal to the actual estimated source signals. That is,
    ``SDR(estimate, reference) - SDR(mix, reference)``.

    For computing ``SDR(estimate, reference)``, PIT (permutation invariant training) is applied,
    so that best combination of sources between the reference signals and the esimate signals
    are picked.

    Args:
        estimate (torch.Tensor): Estimated source signals.
            Shape: [batch, speakers, time frame]
        reference (torch.Tensor): Reference (original) source signals.
            Shape: [batch, speakers, time frame]
        mix (torch.Tensor): Mixed souce signals, from which the setimated signals were generated.
            Shape: [batch, speakers == 1, time frame]

    Returns:
        torch.Tensor: Improved SI-SDR. Shape: [batch, ]
        torch.Tensor: Absolute SI-SDR. Shape: [batch, ]

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """
    estimate = estimate - estimate.mean(axis=2, keepdim=True)
    reference = reference - reference.mean(axis=2, keepdim=True)

    sdr_ = sdr_pit(estimate, reference).unsqueeze(1)
    base_sdr = sdr(mix, reference)  # [batch, speaker]
    return (sdr_ - base_sdr).mean(dim=1), sdr_
