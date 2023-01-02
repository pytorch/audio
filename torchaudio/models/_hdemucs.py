# *****************************************************************************
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *****************************************************************************


import math
import typing as tp
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class _ScaledEmbedding(torch.nn.Module):
    r"""Make continuous embeddings and boost learning rate

    Args:
        num_embeddings (int): number of embeddings
        embedding_dim (int): embedding dimensions
        scale (float, optional): amount to scale learning rate (Default: 10.0)
        smooth (bool, optional): choose to apply smoothing (Default: ``False``)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0, smooth: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, scale raises as sqrt(n), so we normalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for embedding with scale.
        Args:
            x (torch.Tensor): input tensor of shape `(num_embeddings)`

        Returns:
            (Tensor):
                Embedding output of shape `(num_embeddings, embedding_dim)`
        """
        out = self.embedding(x) * self.scale
        return out


class _HEncLayer(torch.nn.Module):

    r"""Encoder layer. This used both by the time and the frequency branch.
    Args:
        chin (int): number of input channels.
        chout (int): number of output channels.
        kernel_size (int, optional): Kernel size for encoder (Default: 8)
        stride (int, optional): Stride for encoder layer (Default: 4)
        norm_groups (int, optional): number of groups for group norm. (Default: 4)
        empty (bool, optional): used to make a layer with just the first conv. this is used
            before merging the time and freq. branches. (Default: ``False``)
        freq (bool, optional): boolean for whether conv layer is for frequency domain (Default: ``True``)
        norm_type (string, optional): Norm type, either ``group_norm `` or ``none`` (Default: ``group_norm``)
        context (int, optional): context size for the 1x1 conv. (Default: 0)
        dconv_kw (Dict[str, Any] or None, optional): dictionary of kwargs for the DConv class. (Default: ``None``)
        pad (bool, optional): true to pad the input. Padding is done so that the output size is
            always the input size / stride. (Default: ``True``)
    """

    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 4,
        empty: bool = False,
        freq: bool = True,
        norm_type: str = "group_norm",
        context: int = 0,
        dconv_kw: Optional[Dict[str, Any]] = None,
        pad: bool = True,
    ):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm_type == "group_norm":
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        pad_val = kernel_size // 4 if pad else 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.pad = pad_val
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad_val = [pad_val, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad_val)
        self.norm1 = norm_fn(chout)

        if self.empty:
            self.rewrite = nn.Identity()
            self.norm2 = nn.Identity()
            self.dconv = nn.Identity()
        else:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)
            self.dconv = _DConv(chout, **dconv_kw)

    def forward(self, x: torch.Tensor, inject: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Forward pass for encoding layer.

        Size depends on whether frequency or time

        Args:
            x (torch.Tensor): tensor input of shape `(B, C, F, T)` for frequency and shape
                `(B, C, T)` for time
            inject (torch.Tensor, optional): on last layer, combine frequency and time branches through inject param,
                same shape as x (default: ``None``)

        Returns:
            Tensor
                output tensor after encoder layer of shape `(B, C, F / stride, T)` for frequency
                    and shape `(B, C, ceil(T / stride))` for time
        """

        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            if inject.shape[-1] != y.shape[-1]:
                raise ValueError("Injection shapes do not align")
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = self.dconv(y)
        z = self.norm2(self.rewrite(y))
        z = F.glu(z, dim=1)
        return z


class _HDecLayer(torch.nn.Module):
    r"""Decoder layer. This used both by the time and the frequency branches.
    Args:
        chin (int): number of input channels.
        chout (int): number of output channels.
        last (bool, optional): whether current layer is final layer (Default: ``False``)
        kernel_size (int, optional): Kernel size for encoder (Default: 8)
        stride (int): Stride for encoder layer (Default: 4)
        norm_groups (int, optional): number of groups for group norm. (Default: 1)
        empty (bool, optional): used to make a layer with just the first conv. this is used
            before merging the time and freq. branches. (Default: ``False``)
        freq (bool, optional): boolean for whether conv layer is for frequency (Default: ``True``)
        norm_type (str, optional): Norm type, either ``group_norm `` or ``none`` (Default: ``group_norm``)
        context (int, optional): context size for the 1x1 conv. (Default: 1)
        dconv_kw (Dict[str, Any] or None, optional): dictionary of kwargs for the DConv class. (Default: ``None``)
        pad (bool, optional): true to pad the input. Padding is done so that the output size is
            always the input size / stride. (Default: ``True``)
    """

    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        norm_type: str = "group_norm",
        context: int = 1,
        dconv_kw: Optional[Dict[str, Any]] = None,
        pad: bool = True,
    ):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm_type == "group_norm":
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            if (kernel_size - stride) % 2 != 0:
                raise ValueError("Kernel size and stride do not align")
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            self.rewrite = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            self.norm1 = norm_fn(2 * chin)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor], length):
        r"""Forward pass for decoding layer.

        Size depends on whether frequency or time

        Args:
            x (torch.Tensor): tensor input of shape `(B, C, F, T)` for frequency and shape
                `(B, C, T)` for time
            skip (torch.Tensor, optional): on first layer, separate frequency and time branches using param
                (default: ``None``)
            length (int): Size of tensor for output

        Returns:
            (Tensor, Tensor):
                Tensor
                    output tensor after decoder layer of shape `(B, C, F * stride, T)` for frequency domain except last
                        frequency layer shape is `(B, C, kernel_size, T)`. Shape is `(B, C, stride * T)`
                        for time domain.
                Tensor
                    contains the output just before final transposed convolution, which is used when the
                        freq. and time branch separate. Otherwise, does not matter. Shape is
                        `(B, C, F, T)` for frequency and `(B, C, T)` for time.
        """
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip
            y = F.glu(self.norm1(self.rewrite(x)), dim=1)
        else:
            y = x
            if skip is not None:
                raise ValueError("Skip must be none when empty is true.")

        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            if z.shape[-1] != length:
                raise ValueError("Last index of z must be equal to length")
        if not self.last:
            z = F.gelu(z)

        return z, y


class HDemucs(torch.nn.Module):
    r"""Hybrid Demucs model from
    *Hybrid Spectrogram and Waveform Source Separation* :cite:`defossez2021hybrid`.

    See Also:
        * :class:`torchaudio.pipelines.SourceSeparationBundle`: Source separation pipeline with pre-trained models.

    Args:
        sources (List[str]): list of source names. List can contain the following source
            options: [``"bass"``, ``"drums"``, ``"other"``, ``"mixture"``, ``"vocals"``].
        audio_channels (int, optional): input/output audio channels. (Default: 2)
        channels (int, optional): initial number of hidden channels. (Default: 48)
        growth (int, optional): increase the number of hidden channels by this factor at each layer. (Default: 2)
        nfft (int, optional): number of fft bins. Note that changing this requires careful computation of
            various shape parameters and will not work out of the box for hybrid models. (Default: 4096)
        depth (int, optional): number of layers in encoder and decoder (Default: 6)
        freq_emb (float, optional): add frequency embedding after the first frequency layer if > 0,
            the actual value controls the weight of the embedding. (Default: 0.2)
        emb_scale (int, optional): equivalent to scaling the embedding learning rate (Default: 10)
        emb_smooth (bool, optional): initialize the embedding with a smooth one (with respect to frequencies).
            (Default: ``True``)
        kernel_size (int, optional): kernel_size for encoder and decoder layers. (Default: 8)
        time_stride (int, optional): stride for the final time layer, after the merge. (Default: 2)
        stride (int, optional): stride for encoder and decoder layers. (Default: 4)
        context (int, optional): context for 1x1 conv in the decoder. (Default: 4)
        context_enc (int, optional): context for 1x1 conv in the encoder. (Default: 0)
        norm_starts (int, optional): layer at which group norm starts being used.
            decoder layers are numbered in reverse order. (Default: 4)
        norm_groups (int, optional): number of groups for group norm. (Default: 4)
        dconv_depth (int, optional): depth of residual DConv branch. (Default: 2)
        dconv_comp (int, optional): compression of DConv branch. (Default: 4)
        dconv_attn (int, optional): adds attention layers in DConv branch starting at this layer. (Default: 4)
        dconv_lstm (int, optional): adds a LSTM layer in DConv branch starting at this layer. (Default: 4)
        dconv_init (float, optional): initial scale for the DConv branch LayerScale. (Default: 1e-4)
    """

    def __init__(
        self,
        sources: List[str],
        audio_channels: int = 2,
        channels: int = 48,
        growth: int = 2,
        nfft: int = 4096,
        depth: int = 6,
        freq_emb: float = 0.2,
        emb_scale: int = 10,
        emb_smooth: bool = True,
        kernel_size: int = 8,
        time_stride: int = 2,
        stride: int = 4,
        context: int = 1,
        context_enc: int = 0,
        norm_starts: int = 4,
        norm_groups: int = 4,
        dconv_depth: int = 2,
        dconv_comp: int = 4,
        dconv_attn: int = 4,
        dconv_lstm: int = 4,
        dconv_init: float = 1e-4,
    ):
        super().__init__()
        self.depth = depth
        self.nfft = nfft
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.channels = channels

        self.hop_length = self.nfft // 4
        self.freq_emb = None

        self.freq_encoder = nn.ModuleList()
        self.freq_decoder = nn.ModuleList()

        self.time_encoder = nn.ModuleList()
        self.time_decoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin * 2  # number of channels for the freq branch
        chout = channels
        chout_z = channels
        freqs = self.nfft // 2

        for index in range(self.depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
            norm_type = "group_norm" if index >= norm_starts else "none"
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                if freqs != 1:
                    raise ValueError("When freq is false, freqs must be 1.")
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm_type": norm_type,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "lstm": lstm,
                    "attn": attn,
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = _HEncLayer(chin_z, chout_z, context=context_enc, **kw)
            if freq:
                if last_freq is True and nfft == 2048:
                    kwt["stride"] = 2
                    kwt["kernel_size"] = 4
                tenc = _HEncLayer(chin, chout, context=context_enc, empty=last_freq, **kwt)
                self.time_encoder.append(tenc)

            self.freq_encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * 2
            dec = _HDecLayer(chout_z, chin_z, last=index == 0, context=context, **kw_dec)
            if freq:
                tdec = _HDecLayer(chout, chin, empty=last_freq, last=index == 0, context=context, **kwt)
                self.time_decoder.insert(0, tdec)
            self.freq_decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = _ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        _rescale_module(self)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        if hl != nfft // 4:
            raise ValueError("Hop length must be nfft // 4")
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = self._pad1d(x, pad, pad + le * hl - x.shape[-1], mode="reflect")

        z = _spectro(x, nfft, hl)[..., :-1, :]
        if z.shape[-1] != le + 4:
            raise ValueError("Spectrogram's last dimension must be 4 + input size divided by stride")
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None):
        hl = self.hop_length
        z = F.pad(z, [0, 0, 0, 1])
        z = F.pad(z, [2, 2])
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = _ispectro(z, hl, length=le)
        x = x[..., pad : pad + length]
        return x

    def _pad1d(self, x: torch.Tensor, padding_left: int, padding_right: int, mode: str = "zero", value: float = 0.0):
        """Wrapper around F.pad, in order for reflect padding when num_frames is shorter than max_pad.
        Add extra zero padding around in order for padding to not break."""
        length = x.shape[-1]
        if mode == "reflect":
            max_pad = max(padding_left, padding_right)
            if length <= max_pad:
                x = F.pad(x, (0, max_pad - length + 1))
        return F.pad(x, (padding_left, padding_right), mode, value)

    def _magnitude(self, z):
        # move the complex dimension to the channel one.
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _mask(self, m):
        # `m` is a full spectrogram and `z` is ignored.
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out

    def forward(self, input: torch.Tensor):

        r"""HDemucs forward call

        Args:
            input (torch.Tensor): input mixed tensor of shape `(batch_size, channel, num_frames)`

        Returns:
            Tensor
                output tensor split into sources of shape `(batch_size, num_sources, channel, num_frames)`
        """

        if input.ndim != 3:
            raise ValueError(f"Expected 3D tensor with dimensions (batch, channel, frames). Found: {input.shape}")

        if input.shape[1] != self.audio_channels:
            raise ValueError(
                f"The channel dimension of input Tensor must match `audio_channels` of HDemucs model. "
                f"Found:{input.shape[1]}."
            )

        x = input
        length = x.shape[-1]

        z = self._spec(input)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = input
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths: List[int] = []  # saved lengths to properly remove padding, freq branch.
        lengths_t: List[int] = []  # saved lengths for time branch.

        for idx, encode in enumerate(self.freq_encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.time_encoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.time_encoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        xt = torch.zeros_like(x)
        # initialize everything to zero (signal will go through u-net skips).

        for idx, decode in enumerate(self.freq_decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.
            offset = self.depth - len(self.time_decoder)
            if idx >= offset:
                tdec = self.time_decoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    if pre.shape[2] != 1:
                        raise ValueError(f"If tdec empty is True, pre shape does not match {pre.shape}")
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        if len(saved) != 0:
            raise AssertionError("saved is not empty")
        if len(lengths_t) != 0:
            raise AssertionError("lengths_t is not empty")
        if len(saved_t) != 0:
            raise AssertionError("saved_t is not empty")

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(x)
        x = self._ispec(zout, length)

        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        return x


class _DConv(torch.nn.Module):
    r"""
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.

    Args:
        channels (int): input/output channels for residual branch.
        compress (float, optional): amount of channel compression inside the branch. (default: 4)
        depth (int, optional): number of layers in the residual branch. Each layer has its own
            projection, and potentially LSTM and attention.(default: 2)
        init (float, optional): initial scale for LayerNorm. (default: 1e-4)
        norm_type (bool, optional): Norm type, either ``group_norm `` or ``none`` (Default: ``group_norm``)
        attn (bool, optional): use LocalAttention. (Default: ``False``)
        heads (int, optional): number of heads for the LocalAttention.  (default: 4)
        ndecay (int, optional): number of decay controls in the LocalAttention. (default: 4)
        lstm (bool, optional): use LSTM. (Default: ``False``)
        kernel_size (int, optional): kernel size for the (dilated) convolutions. (default: 3)
    """

    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        init: float = 1e-4,
        norm_type: str = "group_norm",
        attn: bool = False,
        heads: int = 4,
        ndecay: int = 4,
        lstm: bool = False,
        kernel_size: int = 3,
    ):

        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should not be divisible by 2")
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm_type == "group_norm":
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act = nn.GELU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = pow(2, d) if dilate else 1
            padding = dilation * (kernel_size // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel_size, dilation=dilation, padding=padding),
                norm_fn(hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                nn.GLU(1),
                _LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, _LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, _BLSTM(hidden, layers=2, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        r"""DConv forward call

        Args:
            x (torch.Tensor): input tensor for convolution

        Returns:
            Tensor
                Output after being run through layers.
        """
        for layer in self.layers:
            x = x + layer(x)
        return x


class _BLSTM(torch.nn.Module):
    r"""
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    Args:
        dim (int): dimensions at LSTM layer.
        layers (int, optional): number of LSTM layers. (default: 1)
        skip (bool, optional): (default: ``False``)
    """

    def __init__(self, dim, layers: int = 1, skip: bool = False):
        super().__init__()
        self.max_steps = 200
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""BLSTM forward call

        Args:
            x (torch.Tensor): input tensor for BLSTM shape is `(batch_size, dim, time_steps)`

        Returns:
            Tensor
                Output after being run through bidirectional LSTM. Shape is `(batch_size, dim, time_steps)`
        """
        B, C, T = x.shape
        y = x
        framed = False
        width = 0
        stride = 0
        nframes = 0
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = _unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y

        return x


class _LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).
    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int = 4, ndecay: int = 4):
        r"""
        Args:
            channels (int): Size of Conv1d layers.
            heads (int, optional):  (default: 4)
            ndecay (int, optional): (default: 4)
        """
        super(_LocalState, self).__init__()
        if channels % heads != 0:
            raise ValueError("Channels must be divisible by heads.")
        self.heads = heads
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)

        self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
        if ndecay:
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            if self.query_decay.bias is None:
                raise ValueError("bias must not be None.")
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * 0, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""LocalState forward call

        Args:
            x (torch.Tensor): input tensor for LocalState

        Returns:
            Tensor
                Output after being run through LocalState layer.
        """
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= math.sqrt(keys.shape[2])
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / math.sqrt(self.ndecay)
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class _LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0):
        r"""
        Args:
            channels (int): Size of  rescaling
            init (float, optional): Scale to default to (default: 0)
        """
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""LayerScale forward call

        Args:
            x (torch.Tensor): input tensor for LayerScale

        Returns:
            Tensor
                Output after rescaling tensor.
        """
        return self.scale[:, None] * x


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    shape = list(a.shape[:-1])
    length = int(a.shape[-1])
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(input=a, pad=[0, tgt_length - length])
    strides = [a.stride(dim) for dim in range(a.dim())]
    if strides[-1] != 1:
        raise ValueError("Data should be contiguous.")
    strides = strides[:-1] + [stride, 1]
    shape.append(n_frames)
    shape.append(kernel_size)
    return a.as_strided(shape, strides)


def _rescale_module(module):
    r"""
    Rescales initial weight scale for all models within the module.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            std = sub.weight.std().detach()
            scale = (std / 0.1) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale


def _spectro(x: torch.Tensor, n_fft: int = 512, hop_length: int = 0, pad: int = 0) -> torch.Tensor:
    other = list(x.shape[:-1])
    length = int(x.shape[-1])
    x = x.reshape(-1, length)
    z = torch.stft(
        x,
        n_fft * (1 + pad),
        hop_length,
        window=torch.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frame = z.shape
    other.extend([freqs, frame])
    return z.view(other)


def _ispectro(z: torch.Tensor, hop_length: int = 0, length: int = 0, pad: int = 0) -> torch.Tensor:
    other = list(z.shape[:-2])
    freqs = int(z.shape[-2])
    frames = int(z.shape[-1])

    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = torch.istft(
        z,
        n_fft,
        hop_length,
        window=torch.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, length = x.shape
    other.append(length)
    return x.view(other)


def hdemucs_low(sources: List[str]) -> HDemucs:
    """Builds low nfft (1024) version of :class:`HDemucs`, suitable for sample rates around 8 kHz.

    Args:
        sources (List[str]): See :py:func:`HDemucs`.

    Returns:
        HDemucs:
            HDemucs model.
    """

    return HDemucs(sources=sources, nfft=1024, depth=5)


def hdemucs_medium(sources: List[str]) -> HDemucs:
    r"""Builds medium nfft (2048) version of :class:`HDemucs`, suitable for sample rates of 16-32 kHz.

    .. note::

        Medium HDemucs has not been tested against the original Hybrid Demucs as this nfft and depth configuration is
        not compatible with the original implementation in https://github.com/facebookresearch/demucs

    Args:
        sources (List[str]): See :py:func:`HDemucs`.

    Returns:
        HDemucs:
            HDemucs model.
    """

    return HDemucs(sources=sources, nfft=2048, depth=6)


def hdemucs_high(sources: List[str]) -> HDemucs:
    r"""Builds medium nfft (4096) version of :class:`HDemucs`, suitable for sample rates of 44.1-48 kHz.

    Args:
        sources (List[str]): See :py:func:`HDemucs`.

    Returns:
        HDemucs:
            HDemucs model.
    """

    return HDemucs(sources=sources, nfft=4096, depth=6)
