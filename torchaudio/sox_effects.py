from __future__ import absolute_import, division, print_function, unicode_literals
import torch

import torchaudio


def effect_names():
    """Gets list of valid sox effect names

    Returns: list[str]

    Example
        >>> EFFECT_NAMES = torchaudio.sox_effects.effect_names()
    """
    import _torch_sox
    return _torch_sox.get_effect_names()


def SoxEffect():
    r"""Create an object for passing sox effect information between python and c++

    Returns:
        SoxEffect: An object with the following attributes: ename (str) which is the
        name of effect, and eopts (List[str]) which is a list of effect options.
    """
    import _torch_sox
    return _torch_sox.SoxEffect()


class SoxEffectsChain(object):
    r"""SoX effects chain class.

    Args:
        normalization (bool, number, or callable, optional): If boolean `True`, then output is divided by `1 << 31`
            (assumes signed 32-bit audio), and normalizes to `[-1, 1]`. If `number`, then output is divided by that
            number. If `callable`, then the output is passed as a parameter to the given function, then the
            output is divided by the result. (Default: ``True``)
        channels_first (bool, optional): Set channels first or length first in result.  (Default: ``True``)
        out_siginfo (sox_signalinfo_t, optional): a sox_signalinfo_t type, which could be helpful if the
            audio type cannot be automatically determined. (Default: ``None``)
        out_encinfo (sox_encodinginfo_t, optional): a sox_encodinginfo_t type, which could be set if the
            audio type cannot be automatically determined. (Default: ``None``)
        filetype (str, optional): a filetype or extension to be set if sox cannot determine it
            automatically. . (Default: ``'raw'``)

    Returns:
        Tuple[torch.Tensor, int]: An output Tensor of size `[C x L]` or `[L x C]` where L is the number
        of audio frames and C is the number of channels. An integer which is the sample rate of the
        audio (as listed in the metadata of the file)

    Example
        >>> class MyDataset(Dataset):
        >>>     def __init__(self, audiodir_path):
        >>>         self.data = [os.path.join(audiodir_path, fn) for fn in os.listdir(audiodir_path)]
        >>>         self.E = torchaudio.sox_effects.SoxEffectsChain()
        >>>         self.E.append_effect_to_chain("rate", [16000])  # resample to 16000hz
        >>>         self.E.append_effect_to_chain("channels", ["1"])  # mono signal
        >>>     def __getitem__(self, index):
        >>>         fn = self.data[index]
        >>>         self.E.set_input_file(fn)
        >>>         x, sr = self.E.sox_build_flow_effects()
        >>>         return x, sr
        >>>
        >>>     def __len__(self):
        >>>         return len(self.data)
        >>>
        >>> torchaudio.initialize_sox()
        >>> ds = MyDataset(path_to_audio_files)
        >>> for sig, sr in ds:
        >>>   [do something here]
        >>> torchaudio.shutdown_sox()

    """

    EFFECTS_AVAILABLE = set(effect_names())
    EFFECTS_UNIMPLEMENTED = set(["spectrogram", "splice", "noiseprof", "fir"])

    def __init__(self, normalization=True, channels_first=True, out_siginfo=None, out_encinfo=None, filetype="raw"):
        self.input_file = None
        self.chain = []
        self.MAX_EFFECT_OPTS = 20
        self.out_siginfo = out_siginfo
        self.out_encinfo = out_encinfo
        self.filetype = filetype
        self.normalization = normalization
        self.channels_first = channels_first

    def append_effect_to_chain(self, ename, eargs=None):
        r"""Append effect to a sox effects chain.

        Args:
            ename (str): which is the name of effect
            eargs (List[str]): which is a list of effect options. (Default: ``None``)
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

    def sox_build_flow_effects(self, out=None):
        r"""Build effects chain and flow effects from input file to output tensor

        Args:
            out (torch.Tensor): Where the output will be written to. (Default: ``None``)

        Returns:
            Tuple[torch.Tensor, int]: An output Tensor of size `[C x L]` or `[L x C]` where L is the number
            of audio frames and C is the number of channels. An integer which is the sample rate of the
            audio (as listed in the metadata of the file)
        """
        # initialize output tensor
        if out is not None:
            torchaudio.check_input(out)
        else:
            out = torch.FloatTensor()
        if not len(self.chain):
            e = SoxEffect()
            e.ename = "no_effects"
            e.eopts = [""]
            self.chain.append(e)

        # print("effect options:", [x.eopts for x in self.chain])

        import _torch_sox
        sr = _torch_sox.build_flow_effects(self.input_file,
                                           out,
                                           self.channels_first,
                                           self.out_siginfo,
                                           self.out_encinfo,
                                           self.filetype,
                                           self.chain,
                                           self.MAX_EFFECT_OPTS)

        torchaudio._audio_normalization(out, self.normalization)

        return out, sr

    def clear_chain(self):
        r"""Clear effects chain in python
        """
        self.chain = []

    def set_input_file(self, input_file):
        r"""Set input file for input of chain

        Args:
            input_file (str): The path to the input file.
        """
        self.input_file = input_file

    def _check_effect(self, e):
        if e.lower() in self.EFFECTS_UNIMPLEMENTED:
            raise NotImplementedError("This effect ({}) is not implement in torchaudio".format(e))
        elif e.lower() not in self.EFFECTS_AVAILABLE:
            raise LookupError("Effect name, {}, not valid".format(e.lower()))
        return e.lower()

    # https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
    # convenience function to flatten list recursively
    def _flatten(self, x):
        if x == []:
            return []
        if isinstance(x[0], list):
            return self._flatten(x[:1]) + self._flatten(x[:1])
        return [str(a) for a in x[:1]] + self._flatten(x[1:])
