"""Implementation of MVDR Beamforming Module

Based on https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/beamformer.py

We provide three solutions of MVDR beamforming. One is based on reference channel selection:
Souden, Mehrez, Jacob Benesty, and Sofiene Affes.
"On optimal frequency-domain multichannel linear filtering for noise reduction."
IEEE Transactions on audio, speech, and language processing 18.2 (2009): 260-276.

The other two solutions are based on the steering vector. We apply either eigenvalue decomposition
or the power method to get the steering vector from the PSD matrices.

For eigenvalue decomposistion method, please refer:
Higuchi, Takuya, et al. "Robust MVDR beamforming using time-frequency masks for online/offline ASR in noise."
2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.

For power method, please refer:
Mises, R. V., and Hilda Pollaczek‐Geiringer.
"Praktische Verfahren der Gleichungsauflösung."
ZAMM‐Journal of Applied Mathematics and Mechanics/Zeitschrift für Angewandte Mathematik und Mechanik 9.1 (1929): 58-77.

For online streaming audio, we provide a recursive method to update PSD matrices based on:
Higuchi, Takuya, et al.
"Online MVDR beamformer based on complex Gaussian mixture model with spatial prior for noise robust ASR."
IEEE/ACM Transactions on Audio, Speech, and Language Processing 25.4 (2017): 780-793.
"""

from typing import Optional
import torch


def mat_trace(input: torch.Tensor) -> torch.Tensor:
    r"""Compute the trace of a Tensor

    Args:
        input (torch.Tensor): Tensor of dimension (..., channel, channel)

    Returns:
        torch.Tensor: trace of the input Tensor
    """
    assert input.shape[-1] == input.shape[-2]
    input = torch.diagonal(input, 0, dim1=-1, dim2=-2)
    return input.sum(dim=-1)


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

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            X (torch.Tensor): multi-channel complex-valued STFT matrix.
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

            psd = psd_X * mask.unsqueeze(-1).unsqueeze(-1)
        else:
            psd = psd_X

        psd = psd.sum(dim=-3)
        return psd


class MVDR(torch.nn.Module):
    """Basic MVDR module.

    Args:
        ref_channel (int, optional): the reference channel for beamforming. (Default: ``0``)
        solution (str, optional): the solution to get MVDR weight.
            Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
        diag_loading (bool, optional): whether apply diagonal loading on the psd matrix of noise
            (Default: ``True``)
        diag_eps (float, optional): the coefficient multipied to the identity matrix for diagonal loading
            (Default: 1e-7)
        online (bool, optional): whether to update the mvdr vector based on the previous psd matrices.
            (Default: ``False``)
    """

    def __init__(
        self,
        ref_channel: int = 0,
        solution: Optional[str] = "ref_channel",
        diag_loading: bool = True,
        diag_eps: float = 1e-7,
        online: bool = False,
    ):
        super().__init__()
        assert solution in ["ref_channel", "stv_evd", "stv_power"],\
            "Unknown solution provided. Must be one of [``ref_channel``, ``stv_evd``, ``stv_power``]."
        self.ref_channel = ref_channel
        self.solution = solution
        self.diag_loading = diag_loading
        self.diag_eps = diag_eps
        self.online = online
        self.psd = torch.jit.trace(PSD(), (torch.rand(2, 129, 6, 100, dtype=torch.cdouble), torch.rand(2, 129, 100)))

        psd_s: Optional[torch.Tensor] = torch.zeros(1)
        psd_n: Optional[torch.Tensor] = torch.zeros(1)
        mask_sum_s: Optional[torch.Tensor] = torch.zeros(1)
        mask_sum_n: Optional[torch.Tensor] = torch.zeros(1)
        self.register_buffer('psd_s', psd_s)
        self.register_buffer('psd_n', psd_n)
        self.register_buffer('mask_sum_s', mask_sum_s)
        self.register_buffer('mask_sum_n', mask_sum_n)

    def update_mvdr_vector(
        self,
        psd_s: torch.Tensor,
        psd_n: torch.Tensor,
        mask_s: torch.Tensor,
        mask_n: torch.Tensor,
        reference_vector: torch.Tensor,
        solution: str = 'ref_channel',
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
            solution (str): the solution to estimate the beamforming weight
                (Default: ``ref_channel``)
            diagonal_loading (bool): whether to apply diagonal loading to psd_n
                (Default: ``True``)
            diag_eps (float): The coefficient multipied to the identity matrix for diagonal loading
                (Default: 1e-7)
            eps (float): a value added to the denominator in mask normalization. (Default: 1e-8)

        Returns:
            torch.Tensor: the mvdr beamforming weight matrix
        """
        if self.psd_s == 0:
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = mask_s.sum(dim=-1)
            self.mask_sum_n = mask_n.sum(dim=-1)
            return self.get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)
        else:

            psd_s = self.update_psd_speech(psd_s, mask_s)
            psd_n = self.update_psd_noise(psd_n, mask_n)
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = self.mask_sum_s + mask_s.sum(dim=-1)
            self.mask_sum_n = self.mask_sum_n + mask_n.sum(dim=-1)
            return self.get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)

    def update_psd_speech(self, psd_s: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
        r"""Update psd of speech recursively.

        Args:
            psd_s (torch.Tensor): psd matrix of target speech
            mask_s (torch.Tensor): T-F mask of target speech

        Returns:
            torch.Tensor: the updated psd of speech
        """
        numerator = self.mask_sum_s / (self.mask_sum_s + mask_s.sum(dim=-1))
        denominator = 1 / (self.mask_sum_s + mask_s.sum(dim=-1))
        psd_s = self.psd_s * numerator[..., None, None] + psd_s * denominator[..., None, None]
        return psd_s

    def update_psd_noise(self, psd_n: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
        r"""Update psd of noise recursively.

        Args:
            psd_n (torch.Tensor): psd matrix of target noise
            mask_n (torch.Tensor): T-F mask of target noise

        Returns:
            torch.Tensor: the updated psd of noise
        """
        numerator = self.mask_sum_n / (self.mask_sum_n + mask_n.sum(dim=-1))
        denominator = 1 / (self.mask_sum_n + mask_n.sum(dim=-1))
        psd_n = self.psd_n * numerator[..., None, None] + psd_n * denominator[..., None, None]
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
            solution (str): the solution to estimate the beamforming weight
                (Default: ``ref_channel``)
            diagonal_loading (bool): whether to apply diagonal loading to psd_n
                (Default: ``True``)
            diag_eps (float): The coefficient multipied to the identity matrix for diagonal loading
                (Default: 1e-7)
            eps (float): a value added to the denominator in mask normalization. Default: 1e-8

        Returns:
            torch.Tensor: the mvdr beamforming weight matrix
        """
        if diagonal_loading:
            psd_n = self.tik_reg(psd_n, reg=diag_eps, eps=eps)
        if solution == "ref_channel":
            numerator = torch.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ psd_s
            # ws: (..., C, C) / (...,) -> (..., C, C)
            ws = numerator / (mat_trace(numerator)[..., None, None] + eps)
            # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
            beamform_vector = torch.einsum("...fec,...c->...fe", [ws, reference_vector])
        else:
            if solution == "stv_evd":
                stv = self.get_steering_vector_evd(psd_s)
            else:
                stv = self.get_steering_vector_power(psd_s, psd_n, reference_vector)
            # numerator = psd_n.inv() @ stv
            numerator = torch.linalg.solve(psd_n, stv).squeeze(-1)  # (..., freq, channel)
            # denominator = stv^H @ psd_n.inv() @ stv
            denominator = torch.einsum("...d,...d->...", [stv.conj().squeeze(-1), numerator])
            beamform_vector = numerator / (denominator.real.unsqueeze(-1) + eps)

        return beamform_vector

    def get_steering_vector_evd(self, psd_s: torch.Tensor) -> torch.Tensor:
        r"""Estimate the steering vector by eigenvalue decomposition.

        Args:
            psd_s (torch.tensor): covariance matrix of speech
                Tensor of dimension (..., freq, channel, channel)

        Returns:
            torch.Tensor: the enhanced STFT
                Tensor of dimension (..., freq, channel, 1)
        """
        w, v = torch.linalg.eig(psd_s)  # (..., freq, channel, channel)
        _, indices = torch.max(w.abs(), dim=-1, keepdim=True)
        indices = indices.unsqueeze(-1)
        stv = v.gather(-1, indices.expand(psd_s.shape[:-1] + (1,)))  # (..., freq, channel, 1)
        return stv

    def get_steering_vector_power(
            self,
            psd_s: torch.Tensor,
            psd_n: torch.Tensor,
            reference_vector: torch.Tensor
    ) -> torch.Tensor:
        r"""Estimate the steering vector by the power method.

        Args:
            psd_s (torch.tensor): covariance matrix of speech
                Tensor of dimension (..., freq, channel, channel)
            psd_n (torch.Tensor): covariance matrix of noise
                Tensor of dimension (..., freq, channel, channel)
            reference_vector (torch.Tensor): one-hot reference channel matrix

        Returns:
            torch.Tensor: the enhanced STFT
                Tensor of dimension (..., freq, channel, 1)
        """
        phi = torch.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ stv
        stv = torch.einsum("...fec,...c->...fe", [phi, reference_vector])
        stv = stv.unsqueeze(-1)
        for _ in range(1):
            stv = torch.matmul(phi, stv)
        stv = torch.matmul(psd_s, stv)
        return stv

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
        reg: float = 1e-7,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Perform Tikhonov regularization (only modifying real part).
        Args:
            mat (torch.Tensor): input matrix (..., channel, channel)
            reg (float): regularization factor (Default: 1e-8)
            eps (float): a value to avoid the correlation matrix is all-zero (Default: 1e-8)

        Returns:
            torch.Tensor: regularized matrix (..., channel, channel)
        """
        # Add eps
        C = mat.size(-1)
        eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
        with torch.no_grad():
            epsilon = mat_trace(mat).real[..., None, None] * reg
            # in case that correlation_matrix is all-zero
            epsilon = epsilon + eps
        mat = mat + epsilon * eye[..., :, :]
        return mat

    def forward(self, X: torch.Tensor, mask_s: torch.Tensor, mask_n: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform MVDR beamforming.

        Args:
            X (torch.Tensor): the multi-channel STF of the noisy speech.
                Tensor of dimension (..., channel, freq, time)
            mask_s (torch.Tensor): Time-Frequency mask of target speech
                Tensor of dimension (batch, freq, time)
            mask_n (torch.Tensor, optional): Time-Frequency mask of noise
                Tensor of dimension (batch, freq, time) (Default: None)

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
        mask_s = mask_s.reshape(-1, shape[-2], shape[-1])
        mask_n = mask_n.reshape(-1, shape[-2], shape[-1])

        X = X.transpose(-2, -3)  # (..., freq, channel, time)

        psd_s = self.psd(X, mask_s)
        psd_n = self.psd(X, mask_n)

        u = torch.zeros(
            (X.size()[:-3] + (X.size(-2),)),
            device=X.device,
            dtype=torch.cdouble
        )  # (..., channel)
        u[..., self.ref_channel].fill_(1)

        if self.online:
            w_mvdr = self.update_mvdr_vector(
                psd_s,
                psd_n,
                mask_s,
                mask_n,
                u,
                self.solution,
                self.diag_loading,
                self.diag_eps
            )
        else:
            w_mvdr = self.get_mvdr_vector(
                psd_s,
                psd_n,
                u,
                self.solution,
                self.diag_loading,
                self.diag_eps
            )

        Y = self.apply_beamforming_vector(X, w_mvdr)

        # unpack batch
        Y = Y.reshape(shape[:-3] + Y.shape[-2:])

        return Y
