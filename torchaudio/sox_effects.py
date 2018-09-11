from __future__ import division, print_function
import torch
import _torch_sox

import torchaudio


def effect_names():
    """Gets list of valid sox effect names

    Returns: list[str]

    Example::
        >>> EFFECT_NAMES = torchaudio.sox_effects.effect_names()
    """
    return _torch_sox.get_effect_names()


def SoxEffect():
    """Create an object for passing sox effect information between python and c++

    Returns: SoxEffect(object)
      - ename (str), name of effect
      - eopts (list[str]), list of effect options
    """
    return _torch_sox.SoxEffect()


class SoxEffectsChain(object):
    """SoX effects chain class.
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
        """Append effect to a sox effects chain.
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
        """Build effects chain and flow effects from input file to output tensor
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
        """Clear effects chain in python
        """
        self.chain = []

    def set_input_file(self, input_file):
        """Set input file for input of chain
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
