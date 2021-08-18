"""Implementation of MVDR Beamforming

Based on https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/beamformer.py

We provide two solutions of MVDR beamforming. One is based on reference channel selection:
Souden, Mehrez, Jacob Benesty, and Sofiene Affes.
"On optimal frequency-domain multichannel linear filtering for noise reduction."
IEEE Transactions on audio, speech, and language processing 18.2 (2009): 260-276.

The other solution is based on the steering vector. We apply eigenvalue decomposition to get
the steering vector from the PSD matrix of the target speech.
Higuchi, Takuya, et al. "Robust MVDR beamforming using time-frequency masks for online/offline ASR in noise."
2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.

For online streaming audio, we provide a recursive method to update PSD matrices based on
Higuchi, Takuya, et al.
"Online MVDR beamformer based on complex Gaussian mixture model with spatial prior for noise robust ASR."
IEEE/ACM Transactions on Audio, Speech, and Language Processing 25.4 (2017): 780-793.
"""

from typing import Optional
import torch


def trace(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the trace of a Tensor

    Args:
        input (torch.Tensor): Tensor of dimension (..., channel, channel)

    Returns:
        torch.Tensor: trace of the input Tensor
    """
    assert input.shape[-1] == input.shape[-2]

    shape = list(input.shape)
    strides = list(input.stride())
    strides[-1] += strides[-2]

    shape[-2] = 1
    strides[-2] = 0

    input = torch.as_strided(input, size=shape, stride=strides)
    return input.sum(dim=(-1, -2))


class PSD(torch.nn.Module):
    r"""Compute cross-channel power spectral density (PSD) matrix.

    Args:
        normalize (bool): whether normalize the mask over the time dimension
        eps (float): a value added to the denominator in mask normalization. Default: 1e-15
    """

    def __init__(self, normalize: bool = True, eps: float = 1e-15):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Args:
            X (torch.Tensor): Multi-channel complex-valued STFT matrix.
                Tensor of dimension (..., freq, channel, time)
            mask (torch.Tensor, optional): Time-Frequency mask for normalization.
                Tensor of dimension (..., freq, time)

        Returns:
            psd torch.Tensor: PSD matrix of the input STFT matrix.
                Tensor of dimension (..., freq, channel, channel)
        """
        # outer product: (..., channel_1, time) x (..., channel_2, time) -> (..., time, channel_1, channel_2)
        psd_X = torch.einsum("...ct,...et->...tce", [X, X.conj()])

        if mask is not None:
            # TODO: future support for multi-channel mask

            # Normalized mask along time dimension:
            if self.normalize:
                mask = mask / (mask.sum(dim=-1, keepdim=True) + self.eps)

            psd = psd_X * mask[..., None, None]
        else:
            psd = psd_X

        psd = psd.sum(dim=-3)
        return psd


class MVDR(torch.nn.Module):
    """Basic MVDR module.

    Args:
        ref_channel (int, optional): the reference channel for beamforming. (Default: ``0``)
        solution (str, optional): the solution to get MVDR weight.
            Options: [``ref_channel``, ``rtf``]. (Default: ``ref_channel``)
        diag_loading (bool, optional): whether apply diagonal loading on the psd matrix of noise
        diag_eps (float, optional): the coefficient multipied to the identity matrix for diagonal loading
        online (bool, optional): whether to update the mvdr vector based on the previous psd matrices.
    """

    def __init__(
        self,
        ref_channel: int = 0,
        solution: str = "ref",
        diag_loading: bool = True,
        diag_eps: float = 1e-7,
        online: bool = False,
    ):
        super().__init__()
        self.ref_channel = ref_channel
        self.solution = solution
        self.diag_loading = diag_loading
        self.diag_eps = diag_eps
        self.online = online
        self.psd = PSD()

        if self.online:
            self.psd_s = None
            self.psd_n = None
            self.mask_sum_s = None
            self.mask_sum_n = None

    def update_mvdr_vector(
        self,
        psd_s: torch.Tensor,
        psd_n: torch.Tensor,
        mask_s: torch.Tensor,
        mask_n: torch.Tensor,
        reference_vector: torch.Tensor,
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        r"""Recursively update the MVDR beamforming vector.

        Args:
            psd_s (torch.Tensor): psd matrix of target speech
            psd_n (torch.Tensor): psd matrix of noise
            mask_s (torch.Tensor): T-F mask of target speech
            mask_n (torch.Tensor): T-F mask of noise
            reference_vector (torch.Tensor): one-hot reference channel matrix
            diagonal_loading (bool): whether to apply diagonal loading to psd_n
            diag_eps (float): The coefficient multipied to the identity matrix for diagonal loading
            eps (float): a value added to the denominator in mask normalization. Default: 1e-15

        Returns:
            torch.Tensor: the mvdr beamforming weight matrix
        """
        if self.psd_s is None and self.psd_n is None:
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = mask_s.sum(dim=-1)
            self.mask_sum_n = mask_n.sum(dim=-1)
            return self.get_mvdr_vector(psd_s, psd_n, reference_vector, diagonal_loading, diag_eps, eps)
        else:

            psd_s = self.update_psd_speech(psd_s, mask_s)
            psd_n = self.update_psd_noise(psd_n, mask_n)
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = self.mask_sum_s + mask_s.sum(dim=-1)
            self.mask_sum_n = self.mask_sum_n + mask_n.sum(dim=-1)
            return self.get_mvdr_vector(psd_s, psd_n, reference_vector, diagonal_loading, diag_eps, eps)

    def update_psd_speech(self, psd_s: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
        numerator = self.mask_sum_s / (self.mask_sum_s + mask_s.sum(dim=-1))
        denominator = 1 / (self.mask_sum_s + mask_s.sum(dim=-1))
        psd_s = self.psd_s * numerator[..., None, None] + psd_s * denominator[..., None, None]
        return psd_s

    def update_psd_noise(self, psd_n: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
        numerator = self.mask_sum_n / (self.mask_sum_n + mask_n.sum(dim=-1))
        denominator = 1 / (self.mask_sum_n + mask_n.sum(dim=-1))
        psd_n = self.psd_s * numerator[..., None, None] + psd_n * denominator[..., None, None]
        return psd_n

    def get_mvdr_vector(
        self,
        psd_s: torch.Tensor,
        psd_n: torch.Tensor,
        reference_vector: torch.Tensor,
        solution: str = 'ref_channel',
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        r"""Compute beamforming vector by the reference channel selection method.

        Args:
            psd_s (torch.Tensor): psd matrix of target speech
            psd_n (torch.Tensor): psd matrix of noise
            reference_vector (torch.Tensor): one-hot reference channel matrix
            diagonal_loading (bool): whether to apply diagonal loading to psd_n
            diag_eps (float): The coefficient multipied to the identity matrix for diagonal loading
            eps (float): a value added to the denominator in mask normalization. Default: 1e-15

        Returns:
            torch.Tensor: the mvdr beamforming weight matrix
        """
        if diagonal_loading:
            psd_n = self.tik_reg(psd_n, reg=diag_eps, eps=eps)
        if solution == "ref_channel":
            numerator = torch.linalg.solve(psd_n, psd_s) # psd_n.inv() @ psd_s
            # ws: (..., C, C) / (...,) -> (..., C, C)
            ws = numerator / (trace(numerator)[..., None, None] + eps)
            # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
            beamform_vector = torch.einsum("...fec,...c->...fe", [ws, reference_vector])
        elif solution == "steering_eig":
            w, stv = torch.linalg.eig(psd_s) # (..., freq, channel, channel)
            _, indices = torch.sort(w.abs(), dim=-1, descending=True)
            indices = indices.unsqueeze(-1)
            stv = stv.gather(dim=-1, index=indices)
            # numerator = psd_n.inv() @ stv
            numerator = torch.linalg.solve(psd_n, stv) # (..., freq, channel)
            # denominator = stv^H @ psd_n.inv() @ stv
            denominator = torch.einsum("...d,...d->...", [stv.conj(), numerator])
            beamform_vector = numerator / denominator

        return beamform_vector

    def apply_beamforming_vector(
        self,
        X: torch.Tensor,
        beamform_vector: torch.Tensor
    ) -> torch.Tensor:
        r"""Apply the beamforming weight to the noisy STFT
        Args:
            X (torch.tensor): multi-channel noisy STFT
                Tensor of dimension (..., freq, channel, time)
            beamform_vector (torch.Tensor): beamforming weight matrix
                Tensor of dimension (..., freq, channel)

        Returns:
            torch.Tensor: the enhanced STFT
                Tensor of dimension (..., freq, time)
        """
        # (..., channel) x (..., channel, time) -> (..., time)
        Y = torch.einsum("...c,...ct->...t", [beamform_vector.conj(), X])
        return Y

    def tik_reg(
        self,
        mat: torch.Tensor,
        reg: float = 1e-8,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Perform Tikhonov regularization (only modifying real part).
        Args:
            mat (torch.Tensor): input matrix (..., channel, channel)
            reg (float): regularization factor
            eps (float): a value to avoid the correlation matrix is all-zero

        Returns:
            torch.Tensor: regularized matrix (..., channel, channel)
        """
        # Add eps
        C = mat.size(-1)
        eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
        shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
        eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
        with torch.no_grad():
            epsilon = trace(mat).real[..., None, None] * reg
            # in case that correlation_matrix is all-zero
            epsilon = epsilon + eps
        mat = mat + epsilon * eye
        return mat

    def forward(self, X: torch.Tensor, mask_s: torch.Tensor, mask_n: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform MVDR beamforming.

        Args:
            X (torch.Tensor): The multi-channel STF of the noisy speech.
                Tensor of dimension (..., channel, freq, time)
            mask (torch.Tensor): Tensor of dimension (batch, freq, time)

        Returns:
            torch.Tensor: The single-channel STFT of the enhanced speech.
                Tensor of dimension (..., freq, time)
        """
        if X.ndim < 3:
            raise ValueError(
                f"Expected at least 3D tensor (..., channel, freq, time). Found: {X.shape}"
            )
        if X.dtype != torch.cfloat and X.dtype != torch.cdouble:
            raise ValueError(
                f"The input STFT tensor must be complex-valued. Found: {X.dtype}"
            )
        if X.dtype == torch.cfloat:
            X = X.to(torch.cdouble)
        if mask_n is None:
            mask_n = 1 - mask_s

        shape = X.size()

        # pack batch
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])

        X = X.permute(0, 2, 1, 3)  # (..., freq, channel, time)

        psd_s = self.psd(X, mask_s)
        psd_n = self.psd(X, mask_n)

        u = torch.zeros(
            *(X.size()[:-3] + (X.size(-2),)),
            device=X.device,
            dtype=torch.cdouble
        )
        u[..., self.ref_channel].fill_(1)

        if self.online:
            w_mvdr = self.update_mvdr_vector(psd_s, psd_n, mask_s, mask_n, u)
        else:
            w_mvdr = self.get_mvdr_vector(psd_s, psd_n, u)

        Y = self.apply_beamforming_vector(X, w_mvdr)

        # unpack batch
        Y = Y.reshape(shape[:-3] + Y.shape[-2:])

        return Y
