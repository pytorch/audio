# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Union

import torch
from torch import Tensor
from torchaudio import functional as F


__all__ = []


def _get_mat_trace(input: torch.Tensor, dim1: int = -1, dim2: int = -2) -> torch.Tensor:
    r"""Compute the trace of a Tensor along ``dim1`` and ``dim2`` dimensions.

    Args:
        input (torch.Tensor): Tensor of dimension `(..., channel, channel)`
        dim1 (int, optional): the first dimension of the diagonal matrix
            (Default: -1)
        dim2 (int, optional): the second dimension of the diagonal matrix
            (Default: -2)

    Returns:
        torch.Tensor: trace of the input Tensor
    """
    assert input.ndim >= 2, "The dimension of the tensor must be at least 2."
    assert input.shape[dim1] == input.shape[dim2], "The size of ``dim1`` and ``dim2`` must be the same."
    input = torch.diagonal(input, 0, dim1=dim1, dim2=dim2)
    return input.sum(dim=-1)


class PSD(torch.nn.Module):
    r"""Compute cross-channel power spectral density (PSD) matrix.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        multi_mask (bool, optional): whether to use multi-channel Time-Frequency masks. (Default: ``False``)
        normalize (bool, optional): whether normalize the mask along the time dimension.
        eps (float, optional): a value added to the denominator in mask normalization. (Default: 1e-15)
    """

    def __init__(self, multi_mask: bool = False, normalize: bool = True, eps: float = 1e-15):
        super().__init__()
        self.multi_mask = multi_mask
        self.normalize = normalize
        self.eps = eps

    def forward(self, specgram: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            specgram (torch.Tensor): multi-channel complex-valued STFT matrix.
                Tensor of dimension `(..., channel, freq, time)`
            mask (torch.Tensor or None, optional): Time-Frequency mask for normalization.
                Tensor of dimension `(..., freq, time)` if multi_mask is ``False`` or
                of dimension `(..., channel, freq, time)` if multi_mask is ``True``

        Returns:
            Tensor: PSD matrix of the input STFT matrix.
                Tensor of dimension `(..., freq, channel, channel)`
        """
        # outer product:
        # (..., ch_1, freq, time) x (..., ch_2, freq, time) -> (..., time, ch_1, ch_2)
        psd = torch.einsum("...cft,...eft->...ftce", [specgram, specgram.conj()])

        if mask is not None:
            if self.multi_mask:
                # Averaging mask along channel dimension
                mask = mask.mean(dim=-3)  # (..., freq, time)

            # Normalized mask along time dimension:
            if self.normalize:
                mask = mask / (mask.sum(dim=-1, keepdim=True) + self.eps)

            psd = psd * mask.unsqueeze(-1).unsqueeze(-1)

        psd = psd.sum(dim=-3)
        return psd


class MVDR(torch.nn.Module):
    """Minimum Variance Distortionless Response (MVDR) module that performs MVDR beamforming with Time-Frequency masks.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Based on https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/beamformer.py

    We provide three solutions of MVDR beamforming. One is based on *reference channel selection*
    [:footcite:`souden2009optimal`] (``solution=ref_channel``).

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =\
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bf{\\Phi}_{\\textbf{SS}}}}(f)}\
        {\\text{Trace}({{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f) \\bf{\\Phi}_{\\textbf{SS}}}(f))}}\\bm{u}

    where :math:`\\bf{\\Phi}_{\\textbf{SS}}` and :math:`\\bf{\\Phi}_{\\textbf{NN}}` are the covariance\
        matrices of speech and noise, respectively. :math:`\\bf{u}` is an one-hot vector to determine the\
         reference channel.

    The other two solutions are based on the steering vector (``solution=stv_evd`` or ``solution=stv_power``).

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =\
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}}\
        {{\\bm{v}^{\\mathsf{H}}}(f){\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}

    where :math:`\\bm{v}` is the acoustic transfer function or the steering vector.\
        :math:`.^{\\mathsf{H}}` denotes the Hermitian Conjugate operation.

    We apply either *eigenvalue decomposition*
    [:footcite:`higuchi2016robust`] or the *power method* [:footcite:`mises1929praktische`] to get the
    steering vector from the PSD matrix of speech.

    After estimating the beamforming weight, the enhanced Short-time Fourier Transform (STFT) is obtained by

    .. math::
        \\hat{\\bf{S}} = {\\bf{w}^\\mathsf{H}}{\\bf{Y}}, {\\bf{w}} \\in \\mathbb{C}^{M \\times F}

    where :math:`\\bf{Y}` and :math:`\\hat{\\bf{S}}` are the STFT of the multi-channel noisy speech and\
        the single-channel enhanced speech, respectively.

    For online streaming audio, we provide a *recursive method* [:footcite:`higuchi2017online`] to update the
    PSD matrices of speech and noise, respectively.

    Args:
        ref_channel (int, optional): the reference channel for beamforming. (Default: ``0``)
        solution (str, optional): the solution to get MVDR weight.
            Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
        multi_mask (bool, optional): whether to use multi-channel Time-Frequency masks. (Default: ``False``)
        diag_loading (bool, optional): whether apply diagonal loading on the psd matrix of noise.
            (Default: ``True``)
        diag_eps (float, optional): the coefficient multipied to the identity matrix for diagonal loading.
            (Default: 1e-7)
        online (bool, optional): whether to update the mvdr vector based on the previous psd matrices.
            (Default: ``False``)

    Note:
        To improve the numerical stability, the input spectrogram will be converted to double precision
        (``torch.complex128`` or ``torch.cdouble``) dtype for internal computation. The output spectrogram
        is converted to the dtype of the input spectrogram to be compatible with other modules.

    Note:
        If you use ``stv_evd`` solution, the gradient of the same input may not be identical if the
        eigenvalues of the PSD matrix are not distinct (i.e. some eigenvalues are close or identical).
    """

    def __init__(
        self,
        ref_channel: int = 0,
        solution: str = "ref_channel",
        multi_mask: bool = False,
        diag_loading: bool = True,
        diag_eps: float = 1e-7,
        online: bool = False,
    ):
        super().__init__()
        assert solution in [
            "ref_channel",
            "stv_evd",
            "stv_power",
        ], "Unknown solution provided. Must be one of [``ref_channel``, ``stv_evd``, ``stv_power``]."
        self.ref_channel = ref_channel
        self.solution = solution
        self.multi_mask = multi_mask
        self.diag_loading = diag_loading
        self.diag_eps = diag_eps
        self.online = online
        self.psd = PSD(multi_mask)

        psd_s: torch.Tensor = torch.zeros(1)
        psd_n: torch.Tensor = torch.zeros(1)
        mask_sum_s: torch.Tensor = torch.zeros(1)
        mask_sum_n: torch.Tensor = torch.zeros(1)
        self.register_buffer("psd_s", psd_s)
        self.register_buffer("psd_n", psd_n)
        self.register_buffer("mask_sum_s", mask_sum_s)
        self.register_buffer("mask_sum_n", mask_sum_n)

    def _get_updated_mvdr_vector(
        self,
        psd_s: torch.Tensor,
        psd_n: torch.Tensor,
        mask_s: torch.Tensor,
        mask_n: torch.Tensor,
        reference_vector: torch.Tensor,
        solution: str = "ref_channel",
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
            solution (str, optional): the solution to estimate the beamforming weight
                (Default: ``ref_channel``)
            diagonal_loading (bool, optional): whether to apply diagonal loading to psd_n
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multipied to the identity matrix for diagonal loading
                (Default: 1e-7)
            eps (float, optional): a value added to the denominator in mask normalization. (Default: 1e-8)

        Returns:
            Tensor: the mvdr beamforming weight matrix
        """
        if self.multi_mask:
            # Averaging mask along channel dimension
            mask_s = mask_s.mean(dim=-3)  # (..., freq, time)
            mask_n = mask_n.mean(dim=-3)  # (..., freq, time)
        if self.psd_s.ndim == 1:
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = mask_s.sum(dim=-1)
            self.mask_sum_n = mask_n.sum(dim=-1)
            return self._get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)
        else:
            psd_s = self._get_updated_psd_speech(psd_s, mask_s)
            psd_n = self._get_updated_psd_noise(psd_n, mask_n)
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = self.mask_sum_s + mask_s.sum(dim=-1)
            self.mask_sum_n = self.mask_sum_n + mask_n.sum(dim=-1)
            return self._get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)

    def _get_updated_psd_speech(self, psd_s: torch.Tensor, mask_s: torch.Tensor) -> torch.Tensor:
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

    def _get_updated_psd_noise(self, psd_n: torch.Tensor, mask_n: torch.Tensor) -> torch.Tensor:
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

    def _get_mvdr_vector(
        self,
        psd_s: torch.Tensor,
        psd_n: torch.Tensor,
        reference_vector: torch.Tensor,
        solution: str = "ref_channel",
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        r"""Compute beamforming vector by the reference channel selection method.

        Args:
            psd_s (torch.Tensor): psd matrix of target speech
            psd_n (torch.Tensor): psd matrix of noise
            reference_vector (torch.Tensor): one-hot reference channel matrix
            solution (str, optional): the solution to estimate the beamforming weight
                (Default: ``ref_channel``)
            diagonal_loading (bool, optional): whether to apply diagonal loading to psd_n
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multipied to the identity matrix for diagonal loading
                (Default: 1e-7)
            eps (float, optional): a value added to the denominator in mask normalization. Default: 1e-8

        Returns:
            torch.Tensor: the mvdr beamforming weight matrix
        """
        if diagonal_loading:
            psd_n = self._tik_reg(psd_n, reg=diag_eps, eps=eps)
        if solution == "ref_channel":
            numerator = torch.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ psd_s
            # ws: (..., C, C) / (...,) -> (..., C, C)
            ws = numerator / (_get_mat_trace(numerator)[..., None, None] + eps)
            # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
            beamform_vector = torch.einsum("...fec,...c->...fe", [ws, reference_vector])
        else:
            if solution == "stv_evd":
                stv = self._get_steering_vector_evd(psd_s)
            else:
                stv = self._get_steering_vector_power(psd_s, psd_n, reference_vector)
            # numerator = psd_n.inv() @ stv
            numerator = torch.linalg.solve(psd_n, stv).squeeze(-1)  # (..., freq, channel)
            # denominator = stv^H @ psd_n.inv() @ stv
            denominator = torch.einsum("...d,...d->...", [stv.conj().squeeze(-1), numerator])
            # normalzie the numerator
            scale = stv.squeeze(-1)[..., self.ref_channel, None].conj()
            beamform_vector = numerator / (denominator.real.unsqueeze(-1) + eps) * scale

        return beamform_vector

    def _get_steering_vector_evd(self, psd_s: torch.Tensor) -> torch.Tensor:
        r"""Estimate the steering vector by eigenvalue decomposition.

        Args:
            psd_s (torch.tensor): covariance matrix of speech
                Tensor of dimension `(..., freq, channel, channel)`

        Returns:
            torch.Tensor: the enhanced STFT
                Tensor of dimension `(..., freq, channel, 1)`
        """
        w, v = torch.linalg.eig(psd_s)  # (..., freq, channel, channel)
        _, indices = torch.max(w.abs(), dim=-1, keepdim=True)
        indices = indices.unsqueeze(-1)
        stv = v.gather(-1, indices.expand(psd_s.shape[:-1] + (1,)))  # (..., freq, channel, 1)
        return stv

    def _get_steering_vector_power(
        self, psd_s: torch.Tensor, psd_n: torch.Tensor, reference_vector: torch.Tensor
    ) -> torch.Tensor:
        r"""Estimate the steering vector by the power method.

        Args:
            psd_s (torch.tensor): covariance matrix of speech
                Tensor of dimension `(..., freq, channel, channel)`
            psd_n (torch.Tensor): covariance matrix of noise
                Tensor of dimension `(..., freq, channel, channel)`
            reference_vector (torch.Tensor): one-hot reference channel matrix

        Returns:
            torch.Tensor: the enhanced STFT
                Tensor of dimension `(..., freq, channel, 1)`
        """
        phi = torch.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ psd_s
        stv = torch.einsum("...fec,...c->...fe", [phi, reference_vector])
        stv = stv.unsqueeze(-1)
        stv = torch.matmul(phi, stv)
        stv = torch.matmul(psd_s, stv)
        return stv

    def _apply_beamforming_vector(self, specgram: torch.Tensor, beamform_vector: torch.Tensor) -> torch.Tensor:
        r"""Apply the beamforming weight to the noisy STFT
        Args:
            specgram (torch.tensor): multi-channel noisy STFT
                Tensor of dimension `(..., channel, freq, time)`
            beamform_vector (torch.Tensor): beamforming weight matrix
                Tensor of dimension `(..., freq, channel)`

        Returns:
            torch.Tensor: the enhanced STFT
                Tensor of dimension `(..., freq, time)`
        """
        # (..., channel) x (..., channel, freq, time) -> (..., freq, time)
        specgram_enhanced = torch.einsum("...fc,...cft->...ft", [beamform_vector.conj(), specgram])
        return specgram_enhanced

    def _tik_reg(self, mat: torch.Tensor, reg: float = 1e-7, eps: float = 1e-8) -> torch.Tensor:
        """Perform Tikhonov regularization (only modifying real part).
        Args:
            mat (torch.Tensor): input matrix (..., channel, channel)
            reg (float, optional): regularization factor (Default: 1e-8)
            eps (float, optional): a value to avoid the correlation matrix is all-zero (Default: 1e-8)

        Returns:
            torch.Tensor: regularized matrix (..., channel, channel)
        """
        # Add eps
        C = mat.size(-1)
        eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
        with torch.no_grad():
            epsilon = _get_mat_trace(mat).real[..., None, None] * reg
            # in case that correlation_matrix is all-zero
            epsilon = epsilon + eps
        mat = mat + epsilon * eye[..., :, :]
        return mat

    def forward(
        self, specgram: torch.Tensor, mask_s: torch.Tensor, mask_n: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform MVDR beamforming.

        Args:
            specgram (torch.Tensor): the multi-channel STF of the noisy speech.
                Tensor of dimension `(..., channel, freq, time)`
            mask_s (torch.Tensor): Time-Frequency mask of target speech.
                Tensor of dimension `(..., freq, time)` if multi_mask is ``False``
                or or dimension `(..., channel, freq, time)` if multi_mask is ``True``
            mask_n (torch.Tensor or None, optional): Time-Frequency mask of noise.
                Tensor of dimension `(..., freq, time)` if multi_mask is ``False``
                or or dimension `(..., channel, freq, time)` if multi_mask is ``True``
                (Default: None)

        Returns:
            torch.Tensor: The single-channel STFT of the enhanced speech.
                Tensor of dimension `(..., freq, time)`
        """
        dtype = specgram.dtype
        if specgram.ndim < 3:
            raise ValueError(f"Expected at least 3D tensor (..., channel, freq, time). Found: {specgram.shape}")
        if not specgram.is_complex():
            raise ValueError(
                f"The type of ``specgram`` tensor must be ``torch.cfloat`` or ``torch.cdouble``.\
                    Found: {specgram.dtype}"
            )
        if specgram.dtype == torch.cfloat:
            specgram = specgram.cdouble()  # Convert specgram to ``torch.cdouble``.

        if mask_n is None:
            warnings.warn("``mask_n`` is not provided, use ``1 - mask_s`` as ``mask_n``.")
            mask_n = 1 - mask_s

        shape = specgram.size()

        # pack batch
        specgram = specgram.reshape(-1, shape[-3], shape[-2], shape[-1])
        if self.multi_mask:
            mask_s = mask_s.reshape(-1, shape[-3], shape[-2], shape[-1])
            mask_n = mask_n.reshape(-1, shape[-3], shape[-2], shape[-1])
        else:
            mask_s = mask_s.reshape(-1, shape[-2], shape[-1])
            mask_n = mask_n.reshape(-1, shape[-2], shape[-1])

        psd_s = self.psd(specgram, mask_s)  # (..., freq, time, channel, channel)
        psd_n = self.psd(specgram, mask_n)  # (..., freq, time, channel, channel)

        u = torch.zeros(specgram.size()[:-2], device=specgram.device, dtype=torch.cdouble)  # (..., channel)
        u[..., self.ref_channel].fill_(1)

        if self.online:
            w_mvdr = self._get_updated_mvdr_vector(
                psd_s, psd_n, mask_s, mask_n, u, self.solution, self.diag_loading, self.diag_eps
            )
        else:
            w_mvdr = self._get_mvdr_vector(psd_s, psd_n, u, self.solution, self.diag_loading, self.diag_eps)

        specgram_enhanced = self._apply_beamforming_vector(specgram, w_mvdr)

        # unpack batch
        specgram_enhanced = specgram_enhanced.reshape(shape[:-3] + shape[-2:])

        return specgram_enhanced.to(dtype)


class RTFMVDR(torch.nn.Module):
    r"""Minimum Variance Distortionless Response (*MVDR* [:footcite:`capon1969high`]) module
    based on the relative transfer function (RTF) and power spectral density (PSD) matrix of noise.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the multi-channel complex-valued spectrum :math:`\textbf{Y}`, the relative transfer function (RTF) matrix
    or the steering vector of target speech :math:`\bm{v}`, the PSD matrix of noise :math:`\bf{\Phi}_{\textbf{NN}}`, and
    a one-hot vector that represents the reference channel :math:`\bf{u}`, the module computes the single-channel
    complex-valued spectrum of the enhanced speech :math:`\hat{\textbf{S}}`. The formula is defined as:

    .. math::
        \hat{\textbf{S}}(f) = \textbf{w}_{\text{bf}}(f)^{\mathsf{H}} \textbf{Y}(f)

    where :math:`\textbf{w}_{\text{bf}}(f)` is the MVDR beamforming weight for the :math:`f`-th frequency bin,
    :math:`(.)^{\mathsf{H}}` denotes the Hermitian Conjugate operation.

    The beamforming weight is computed by:

    .. math::
        \textbf{w}_{\text{MVDR}}(f) =
        \frac{{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bm{v}}(f)}}
        {{\bm{v}^{\mathsf{H}}}(f){\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bm{v}}(f)}
    """

    def forward(
        self,
        specgram: Tensor,
        rtf: Tensor,
        psd_n: Tensor,
        reference_channel: Union[int, Tensor],
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Args:
            specgram (torch.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`
            rtf (torch.Tensor): The complex-valued RTF vector of target speech.
                Tensor with dimensions `(..., freq, channel)`.
            psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            reference_channel (int or torch.Tensor): Specifies the reference channel.
                If the dtype is ``int``, it represents the reference channel index.
                If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
                is one-hot.
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            torch.Tensor: Single-channel complex-valued enhanced spectrum with dimensions `(..., freq, time)`.
        """
        w_mvdr = F.mvdr_weights_rtf(rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps)
        spectrum_enhanced = F.apply_beamforming(w_mvdr, specgram)
        return spectrum_enhanced


class SoudenMVDR(torch.nn.Module):
    r"""Minimum Variance Distortionless Response (*MVDR* [:footcite:`capon1969high`]) module
    based on the method proposed by *Souden et, al.* [:footcite:`souden2009optimal`].

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the multi-channel complex-valued spectrum :math:`\textbf{Y}`, the power spectral density (PSD) matrix
    of target speech :math:`\bf{\Phi}_{\textbf{SS}}`, the PSD matrix of noise :math:`\bf{\Phi}_{\textbf{NN}}`, and
    a one-hot vector that represents the reference channel :math:`\bf{u}`, the module computes the single-channel
    complex-valued spectrum of the enhanced speech :math:`\hat{\textbf{S}}`. The formula is defined as:

    .. math::
        \hat{\textbf{S}}(f) = \textbf{w}_{\text{bf}}(f)^{\mathsf{H}} \textbf{Y}(f)

    where :math:`\textbf{w}_{\text{bf}}(f)` is the MVDR beamforming weight for the :math:`f`-th frequency bin.

    The beamforming weight is computed by:

    .. math::
        \textbf{w}_{\text{MVDR}}(f) =
        \frac{{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bf{\Phi}_{\textbf{SS}}}}(f)}
        {\text{Trace}({{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f) \bf{\Phi}_{\textbf{SS}}}(f))}}\bm{u}
    """

    def forward(
        self,
        specgram: Tensor,
        psd_s: Tensor,
        psd_n: Tensor,
        reference_channel: Union[int, Tensor],
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Args:
            specgram (torch.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`.
            psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            reference_channel (int or torch.Tensor): Specifies the reference channel.
                If the dtype is ``int``, it represents the reference channel index.
                If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
                is one-hot.
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            torch.Tensor: Single-channel complex-valued enhanced spectrum with dimensions `(..., freq, time)`.
        """
        w_mvdr = F.mvdr_weights_souden(psd_s, psd_n, reference_channel, diagonal_loading, diag_eps, eps)
        spectrum_enhanced = F.apply_beamforming(w_mvdr, specgram)
        return spectrum_enhanced
