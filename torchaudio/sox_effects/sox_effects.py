import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torchaudio._internal import (
    module_utils as _mod_utils,
    misc_ops as _misc_ops,
)
from torchaudio.utils.sox_utils import list_effects


if _mod_utils.is_module_available('torchaudio._torchaudio'):
    from torchaudio import _torchaudio


@_mod_utils.requires_module('torchaudio._torchaudio')
def init_sox_effects():
    """Initialize resources required to use sox effects.

    Note:
        You do not need to call this function manually. It is called automatically.

    Once initialized, you do not need to call this function again across the multiple uses of
    sox effects though it is safe to do so as long as :func:`shutdown_sox_effects` is not called yet.
    Once :func:`shutdown_sox_effects` is called, you can no longer use SoX effects and initializing
    again will result in error.
    """
    torch.ops.torchaudio.sox_effects_initialize_sox_effects()


@_mod_utils.requires_module("torchaudio._torchaudio")
def shutdown_sox_effects():
    """Clean up resources required to use sox effects.

    Note:
        You do not need to call this function manually. It is called automatically.

    It is safe to call this function multiple times.
    Once :py:func:`shutdown_sox_effects` is called, you can no longer use SoX effects and
    initializing again will result in error.
    """
    torch.ops.torchaudio.sox_effects_shutdown_sox_effects()


@_mod_utils.requires_module('torchaudio._torchaudio')
def effect_names() -> List[str]:
    """Gets list of valid sox effect names

    Returns:
        List[str]: list of available effect names.

    Example
        >>> torchaudio.sox_effects.effect_names()
        ['allpass', 'band', 'bandpass', ... ]
    """
    return list(list_effects().keys())


@_mod_utils.requires_module('torchaudio._torchaudio')
def apply_effects_tensor(
        tensor: torch.Tensor,
        sample_rate: int,
        effects: List[List[str]],
        channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Apply sox effects to given Tensor

    Note:
        This function works in the way very similar to ``sox`` command, however there are slight
        differences. For example, ``sox`` commnad adds certain effects automatically (such as
        ``rate`` effect after ``speed`` and ``pitch`` and other effects), but this function does
        only applies the given effects. (Therefore, to actually apply ``speed`` effect, you also
        need to give ``rate`` effect with desired sampling rate.)

    Args:
        tensor (torch.Tensor): Input 2D Tensor.
        sample_rate (int): Sample rate
        effects (List[List[str]]): List of effects.
        channels_first (bool): Indicates if the input Tensor's dimension is
            ``[channels, time]`` or ``[time, channels]``

    Returns:
        Tuple[torch.Tensor, int]: Resulting Tensor and sample rate.
        The resulting Tensor has the same ``dtype`` as the input Tensor, and
        the same channels order. The shape of the Tensor can be different based on the
        effects applied. Sample rate can also be different based on the effects applied.

    Example - Basic usage
        >>>
        >>> # Defines the effects to apply
        >>> effects = [
        ...     ['gain', '-n'],  # normalises to 0dB
        ...     ['pitch', '5'],  # 5 cent pitch shift
        ...     ['rate', '8000'],  # resample to 8000 Hz
        ... ]
        >>>
        >>> # Generate pseudo wave:
        >>> # normalized, channels first, 2ch, sampling rate 16000, 1 second
        >>> sample_rate = 16000
        >>> waveform = 2 * torch.rand([2, sample_rate * 1]) - 1
        >>> waveform.shape
        torch.Size([2, 16000])
        >>> waveform
        tensor([[ 0.3138,  0.7620, -0.9019,  ..., -0.7495, -0.4935,  0.5442],
                [-0.0832,  0.0061,  0.8233,  ..., -0.5176, -0.9140, -0.2434]])
        >>>
        >>> # Apply effects
        >>> waveform, sample_rate = apply_effects_tensor(
        ...     wave_form, sample_rate, effects, channels_first=True)
        >>>
        >>> # Check the result
        >>> # The new waveform is sampling rate 8000, 1 second.
        >>> # normalization and channel order are preserved
        >>> waveform.shape
        torch.Size([2, 8000])
        >>> waveform
        tensor([[ 0.5054, -0.5518, -0.4800,  ..., -0.0076,  0.0096, -0.0110],
                [ 0.1331,  0.0436, -0.3783,  ..., -0.0035,  0.0012,  0.0008]])
        >>> sample_rate
        8000

    Example - Torchscript-able transform
        >>>
        >>> # Use `apply_effects_tensor` in `torch.nn.Module` and dump it to file,
        >>> # then run sox effect via Torchscript runtime.
        >>>
        >>> class SoxEffectTransform(torch.nn.Module):
        ...     effects: List[List[str]]
        ...
        ...     def __init__(self, effects: List[List[str]]):
        ...         super().__init__()
        ...         self.effects = effects
        ...
        ...     def forward(self, tensor: torch.Tensor, sample_rate: int):
        ...         return sox_effects.apply_effects_tensor(
        ...             tensor, sample_rate, self.effects)
        ...
        ...
        >>> # Create transform object
        >>> effects = [
        ...     ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
        ...     ["rate", "8000"],  # change sample rate to 8000
        ... ]
        >>> transform = SoxEffectTensorTransform(effects, input_sample_rate)
        >>>
        >>> # Dump it to file and load
        >>> path = 'sox_effect.zip'
        >>> torch.jit.script(trans).save(path)
        >>> transform = torch.jit.load(path)
        >>>
        >>>> # Run transform
        >>> waveform, input_sample_rate = torchaudio.load("input.wav")
        >>> waveform, sample_rate = transform(waveform, input_sample_rate)
        >>> assert sample_rate == 8000
    """
    in_signal = torch.classes.torchaudio.TensorSignal(tensor, sample_rate, channels_first)
    out_signal = torch.ops.torchaudio.sox_effects_apply_effects_tensor(in_signal, effects)
    return out_signal.get_tensor(), out_signal.get_sample_rate()


@_mod_utils.requires_module('torchaudio._torchaudio')
def apply_effects_file(
        path: str,
        effects: List[List[str]],
        normalize: bool = True,
        channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Apply sox effects to the audio file and load the resulting data as Tensor

    Note:
        This function works in the way very similar to ``sox`` command, however there are slight
        differences. For example, ``sox`` commnad adds certain effects automatically (such as
        ``rate`` effect after ``speed``, ``pitch`` etc), but this function only applies the given
        effects. Therefore, to actually apply ``speed`` effect, you also need to give ``rate``
        effect with desired sampling rate, because internally, ``speed`` effects only alter sampling
        rate and leave samples untouched.

    Args:
        path (str): Path to the audio file.
        effects (List[List[str]]): List of effects.
        normalize (bool):
            When ``True``, this function always return ``float32``, and sample values are
            normalized to ``[-1.0, 1.0]``.
            If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
            integer type. This argument has no effect for formats other
            than integer WAV type.
        channels_first (bool): When True, the returned Tensor has dimension ``[channel, time]``.
            Otherwise, the returned Tensor's dimension is ``[time, channel]``.

    Returns:
        Tuple[torch.Tensor, int]: Resulting Tensor and sample rate.
        If ``normalize=True``, the resulting Tensor is always ``float32`` type.
        If ``normalize=False`` and the input audio file is of integer WAV file, then the
        resulting Tensor has corresponding integer type. (Note 24 bit integer type is not supported)
        If ``channels_first=True``, the resulting Tensor has dimension ``[channel, time]``,
        otherwise ``[time, channel]``.

    Example - Basic usage
        >>>
        >>> # Defines the effects to apply
        >>> effects = [
        ...     ['gain', '-n'],  # normalises to 0dB
        ...     ['pitch', '5'],  # 5 cent pitch shift
        ...     ['rate', '8000'],  # resample to 8000 Hz
        ... ]
        >>>
        >>> # Apply effects and load data with channels_first=True
        >>> waveform, sample_rate = apply_effects_file("data.wav", effects, channels_first=True)
        >>>
        >>> # Check the result
        >>> waveform.shape
        torch.Size([2, 8000])
        >>> waveform
        tensor([[ 5.1151e-03,  1.8073e-02,  2.2188e-02,  ...,  1.0431e-07,
                 -1.4761e-07,  1.8114e-07],
                [-2.6924e-03,  2.1860e-03,  1.0650e-02,  ...,  6.4122e-07,
                 -5.6159e-07,  4.8103e-07]])
        >>> sample_rate
        8000

    Example - Apply random speed perturbation to dataset
        >>>
        >>> # Load data from file, apply random speed perturbation
        >>> class RandomPerturbationFile(torch.utils.data.Dataset):
        ...     \"\"\"Given flist, apply random speed perturbation
        ...
        ...     Suppose all the input files are at least one second long.
        ...     \"\"\"
        ...     def __init__(self, flist: List[str], sample_rate: int):
        ...         super().__init__()
        ...         self.flist = flist
        ...         self.sample_rate = sample_rate
        ...         self.rng = None
        ...
        ...     def __getitem__(self, index):
        ...         speed = self.rng.uniform(0.5, 2.0)
        ...         effects = [
        ...             ['gain', '-n', '-10'],  # apply 10 db attenuation
        ...             ['remix', '-'],  # merge all the channels
        ...             ['speed', f'{speed:.5f}'],  # duration is now 0.5 ~ 2.0 seconds.
        ...             ['rate', f'{self.sample_rate}'],
        ...             ['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
        ...             ['trim', '0', '2'],  # get the first 2 seconds
        ...         ]
        ...         waveform, _ = torchaudio.sox_effects.apply_effects_file(
        ...             self.flist[index], effects)
        ...         return waveform
        ...
        ...     def __len__(self):
        ...         return len(self.flist)
        ...
        >>> dataset = RandomPerturbationFile(file_list, sample_rate=8000)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        >>>     pass
    """
    signal = torch.ops.torchaudio.sox_effects_apply_effects_file(path, effects, normalize, channels_first)
    return signal.get_tensor(), signal.get_sample_rate()


@_mod_utils.requires_module('torchaudio._torchaudio')
@_mod_utils.deprecated('Please migrate to `apply_effects_file` or `apply_effects_tensor`.', "0.8.0")
def SoxEffect():
    r"""Create an object for passing sox effect information between python and c++

    Warning:
        This function is deprecated.
        Please migrate to :func:`apply_effects_file` or :func:`apply_effects_tensor`.

    Returns:
        SoxEffect: An object with the following attributes: ename (str) which is the
        name of effect, and eopts (List[str]) which is a list of effect options.
    """
    return _torchaudio.SoxEffect()


class SoxEffectsChain(object):
    r"""SoX effects chain class.

    Warning:
        This class is deprecated.
        Please migrate to :func:`apply_effects_file` or :func:`apply_effects_tensor`.

    Args:
        normalization (bool, number, or callable, optional):
            If boolean ``True``, then output is divided by ``1 << 31``
            (assumes signed 32-bit audio), and normalizes to ``[-1, 1]``.
            If ``number``, then output is divided by that number.
            If ``callable``, then the output is passed as a parameter to the given function, then
            the output is divided by the result. (Default: ``True``)
        channels_first (bool, optional):
            Set channels first or length first in result.  (Default: ``True``)
        out_siginfo (sox_signalinfo_t, optional):
            a sox_signalinfo_t type, which could be helpful if the audio type cannot be
            automatically determined. (Default: ``None``)
        out_encinfo (sox_encodinginfo_t, optional):
            a sox_encodinginfo_t type, which could be set if the audio type cannot be
            automatically determined. (Default: ``None``)
        filetype (str, optional):
            a filetype or extension to be set if sox cannot determine it automatically.
            (Default: ``'raw'``)

    Returns:
        Tuple[Tensor, int]:
        An output Tensor of size ``[C x L]`` or ``[L x C]`` where L is the number
        of audio frames and C is the number of channels. An integer which is the sample rate of the
        audio (as listed in the metadata of the file)

    Example
        >>> class MyDataset(Dataset):
        ...     def __init__(self, audiodir_path):
        ...         self.data = [
        ...             os.path.join(audiodir_path, fn)
        ...             for fn in os.listdir(audiodir_path)]
        ...         self.E = torchaudio.sox_effects.SoxEffectsChain()
        ...         self.E.append_effect_to_chain("rate", [16000])  # resample to 16000hz
        ...         self.E.append_effect_to_chain("channels", ["1"])  # mono signal
        ...     def __getitem__(self, index):
        ...         fn = self.data[index]
        ...         self.E.set_input_file(fn)
        ...         x, sr = self.E.sox_build_flow_effects()
        ...         return x, sr
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        ...
        >>> ds = MyDataset(path_to_audio_files)
        >>> for sig, sr in ds:
        ...    pass
    """

    EFFECTS_UNIMPLEMENTED = {"spectrogram", "splice", "noiseprof", "fir"}

    def __init__(self,
                 normalization: Union[bool, float, Callable] = True,
                 channels_first: bool = True,
                 out_siginfo: Any = None,
                 out_encinfo: Any = None,
                 filetype: str = "raw") -> None:
        warnings.warn(
            'torchaudio.sox_effects.SoxEffectsChain has been deprecated and '
            'will be removed from 0.8.0 release. '
            'Please migrate to `apply_effects_file` or `apply_effects_tensor`.'
        )
        self.input_file: Optional[str] = None
        self.chain: List[str] = []
        self.MAX_EFFECT_OPTS = 20
        self.out_siginfo = out_siginfo
        self.out_encinfo = out_encinfo
        self.filetype = filetype
        self.normalization = normalization
        self.channels_first = channels_first

        # Define in __init__ to avoid calling at import time
        self.EFFECTS_AVAILABLE = set(effect_names())

    def append_effect_to_chain(self,
                               ename: str,
                               eargs: Optional[Union[List[str], str]] = None) -> None:
        r"""Append effect to a sox effects chain.

        Args:
            ename (str): which is the name of effect
            eargs (List[str] or str, optional): which is a list of effect options. (Default: ``None``)
        """
        e = SoxEffect()
        # check if we have a valid effect
        ename = self._check_effect(ename)
        if eargs is None or eargs == []:
            eargs = [""]
        elif not isinstance(eargs, list):
            eargs = [eargs]
        eargs = self._flatten(eargs)
        if len(eargs) > self.MAX_EFFECT_OPTS:
            raise RuntimeError("Number of effect options ({}) is greater than max "
                               "suggested number of options {}.  Increase MAX_EFFECT_OPTS "
                               "or lower the number of effect options".format(len(eargs), self.MAX_EFFECT_OPTS))
        e.ename = ename
        e.eopts = eargs
        self.chain.append(e)

    @_mod_utils.requires_module('torchaudio._torchaudio')
    def sox_build_flow_effects(self,
                               out: Optional[Tensor] = None) -> Tuple[Tensor, int]:
        r"""Build effects chain and flow effects from input file to output tensor

        Args:
            out (Tensor, optional): Where the output will be written to. (Default: ``None``)

        Returns:
            Tuple[Tensor, int]: An output Tensor of size `[C x L]` or `[L x C]` where
            L is the number of audio frames and C is the number of channels.
            An integer which is the sample rate of the audio (as listed in the metadata of the file)
        """
        # initialize output tensor
        if out is not None:
            _misc_ops.check_input(out)
        else:
            out = torch.FloatTensor()
        if not len(self.chain):
            e = SoxEffect()
            e.ename = "no_effects"
            e.eopts = [""]
            self.chain.append(e)

        # print("effect options:", [x.eopts for x in self.chain])

        sr = _torchaudio.build_flow_effects(self.input_file,
                                            out,
                                            self.channels_first,
                                            self.out_siginfo,
                                            self.out_encinfo,
                                            self.filetype,
                                            self.chain,
                                            self.MAX_EFFECT_OPTS)

        _misc_ops.normalize_audio(out, self.normalization)

        return out, sr

    def clear_chain(self) -> None:
        r"""Clear effects chain in python
        """
        self.chain = []

    def set_input_file(self, input_file: str) -> None:
        r"""Set input file for input of chain

        Args:
            input_file (str): The path to the input file.
        """
        self.input_file = input_file

    def _check_effect(self, e: str) -> str:
        if e.lower() in self.EFFECTS_UNIMPLEMENTED:
            raise NotImplementedError("This effect ({}) is not implement in torchaudio".format(e))
        elif e.lower() not in self.EFFECTS_AVAILABLE:
            raise LookupError("Effect name, {}, not valid".format(e.lower()))
        return e.lower()

    # https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
    # convenience function to flatten list recursively
    def _flatten(self, x: list) -> list:
        if x == []:
            return []
        if isinstance(x[0], list):
            return self._flatten(x[:1]) + self._flatten(x[:1])
        return [str(a) for a in x[:1]] + self._flatten(x[1:])
