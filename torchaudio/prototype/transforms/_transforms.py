from typing import Callable, Optional

import torch
from torchaudio.prototype.functional import barkscale_fbanks
from torchaudio.transforms import Spectrogram


class BarkScale(torch.nn.Module):
    r"""Turn a normal STFT into a bark frequency STFT with triangular filter banks.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        n_barks (int, optional): Number of bark filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If ``"slaney"``, divide the triangular bark weights by the width of the bark band
            (area normalization). (Default: ``None``)
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> barkscale_transform = transforms.BarkScale(sample_rate=sample_rate, n_stft=1024 // 2 + 1)
        >>> barkscale_spectrogram = barkscale_transform(spectrogram)

    See also:
        :py:func:`torchaudio.prototype.functional.barkscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ["n_barks", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_barks: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        bark_scale: str = "traunmuller",
    ) -> None:
        super(BarkScale, self).__init__()
        self.n_barks = n_barks
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.bark_scale = bark_scale

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))

        fb = barkscale_fbanks(n_stft, self.f_min, self.f_max, self.n_barks, self.sample_rate, self.bark_scale)
        self.register_buffer("fb", fb)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            specgram (torch.Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            torch.Tensor: Bark frequency spectrogram of size (..., ``n_barks``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        bark_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return bark_specgram


class InverseBarkScale(torch.nn.Module):
    r"""Estimate a STFT in normal frequency domain from bark frequency domain.

    .. devices:: CPU CUDA

    It minimizes the euclidian norm between the input bark-spectrogram and the product between
    the estimated spectrogram and the filter banks using SGD.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_barks (int, optional): Number of bark filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        max_iter (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
        tolerance_loss (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
        tolerance_change (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
        sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> mel_spectrogram_transform = transforms.BarkSpectrogram(sample_rate, n_fft=1024)
        >>> mel_spectrogram = bark_spectrogram_transform(waveform)
        >>> inverse_barkscale_transform = transforms.InverseBarkScale(n_stft=1024 // 2 + 1)
        >>> spectrogram = inverse_barkscale_transform(mel_spectrogram)
    """
    __constants__ = [
        "n_stft",
        "n_barks",
        "sample_rate",
        "f_min",
        "f_max",
        "max_iter",
        "tolerance_loss",
        "tolerance_change",
        "sgdargs",
    ]

    def __init__(
        self,
        n_stft: int,
        n_barks: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        max_iter: int = 100000,
        tolerance_loss: float = 1e-5,
        tolerance_change: float = 1e-8,
        sgdargs: Optional[dict] = None,
        bark_scale: str = "traunmuller",
    ) -> None:
        super(InverseBarkScale, self).__init__()
        self.n_barks = n_barks
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.max_iter = max_iter
        self.tolerance_loss = tolerance_loss
        self.tolerance_change = tolerance_change
        self.sgdargs = sgdargs or {"lr": 0.1, "momentum": 0.9}

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))

        fb = barkscale_fbanks(n_stft, self.f_min, self.f_max, self.n_barks, self.sample_rate, bark_scale)
        self.register_buffer("fb", fb)

    def forward(self, barkspec: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            barkspec (torch.Tensor): A Bark frequency spectrogram of dimension (..., ``n_barks``, time)

        Returns:
            torch.Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        # pack batch
        shape = barkspec.size()
        barkspec = barkspec.view(-1, shape[-2], shape[-1])

        n_barks, time = shape[-2], shape[-1]
        freq, _ = self.fb.size()  # (freq, n_mels)
        barkspec = barkspec.transpose(-1, -2)
        if self.n_barks != n_barks:
            raise ValueError("Expected an input with {} bark bins. Found: {}".format(self.n_barks, n_barks))

        specgram = torch.rand(
            barkspec.size()[0], time, freq, requires_grad=True, dtype=barkspec.dtype, device=barkspec.device
        )

        optim = torch.optim.SGD([specgram], **self.sgdargs)

        loss = float("inf")
        for _ in range(self.max_iter):
            optim.zero_grad()
            diff = barkspec - specgram.matmul(self.fb)
            new_loss = diff.pow(2).sum(axis=-1).mean()
            # take sum over bark-frequency then average over other dimensions
            # so that loss threshold is applied par unit timeframe
            new_loss.backward()
            optim.step()
            specgram.data = specgram.data.clamp(min=0)

            new_loss = new_loss.item()
            if new_loss < self.tolerance_loss or abs(loss - new_loss) < self.tolerance_change:
                break
            loss = new_loss

        specgram.requires_grad_(False)
        specgram = specgram.clamp(min=0).transpose(-1, -2)

        # unpack batch
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram


class BarkSpectrogram(torch.nn.Module):
    r"""Create BarkSpectrogram for a raw audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This is a composition of :py:func:`torchaudio.transforms.Spectrogram` and
    and :py:func:`torchaudio.transforms.BarkScale`.

    Sources
        * https://www.fon.hum.uva.nl/praat/manual/BarkSpectrogram.html
        * Traunmüller, Hartmut. "Analytical Expressions for the Tonotopic Sensory Scale." Journal of the Acoustical
        * Society of America. Vol. 88, Issue 1, 1990, pp. 97–100.
        * https://ccrma.stanford.edu/courses/120-fall-2003/lecture-5.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., torch.Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.BarkSpectrogram(sample_rate)
        >>> bark_specgram = transform(waveform)  # (channel, n_barks, time)

    See also:
        :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
        generate the filter banks.
    """
    __constants__ = ["sample_rate", "n_fft", "win_length", "hop_length", "pad", "n_barks", "f_min"]

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_barks: int = 128,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        bark_scale: str = "traunmuller",
    ) -> None:
        super(BarkSpectrogram, self).__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_barks = n_barks  # number of bark frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.bark_scale = BarkScale(
            self.n_barks, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, bark_scale
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (torch.Tensor): torch.Tensor of audio of dimension (..., time).

        Returns:
            torch.Tensor: Bark frequency spectrogram of size (..., ``n_barks``, time).
        """
        specgram = self.spectrogram(waveform)
        bark_specgram = self.bark_scale(specgram)
        return bark_specgram
