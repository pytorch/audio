import torch
import _torch_sox

import torchaudio

EFFECT_NAMES = set(_torch_sox.get_effect_names())

"""
Notes:

sox_signalinfo_t {
  sox_rate_t       rate;         /**< samples per second, 0 if unknown */
  unsigned         channels;     /**< number of sound channels, 0 if unknown */
  unsigned         precision;    /**< bits per sample, 0 if unknown */
  sox_uint64_t     length;       /**< samples * chans in file, 0 if unspecified, -1 if unknown */
  double           * mult;       /**< Effects headroom multiplier; may be null */
}

typedef struct sox_encodinginfo_t {
  sox_encoding_t encoding; /**< format of sample numbers */
  unsigned bits_per_sample;  /**< 0 if unknown or variable; uncompressed value if lossless; compressed value if lossy */
  double compression;      /**< compression factor (where applicable) */
  sox_option_t reverse_bytes;  /** use sox_option_default */
  sox_option_t reverse_nibbles;  /** use sox_option_default */
  sox_option_t reverse_bits;  /** use sox_option_default */
  sox_bool opposite_endian;  /** use sox_false */
}

sox_encodings_t = {
  "SOX_ENCODING_UNKNOWN",
  "SOX_ENCODING_SIGN2",
  "SOX_ENCODING_UNSIGNED",
  "SOX_ENCODING_FLOAT",
  "SOX_ENCODING_FLOAT_TEXT",
  "SOX_ENCODING_FLAC",
  "SOX_ENCODING_HCOM",
  "SOX_ENCODING_WAVPACK",
  "SOX_ENCODING_WAVPACKF",
  "SOX_ENCODING_ULAW",
  "SOX_ENCODING_ALAW",
  "SOX_ENCODING_G721",
  "SOX_ENCODING_G723",
  "SOX_ENCODING_CL_ADPCM",
  "SOX_ENCODING_CL_ADPCM16",
  "SOX_ENCODING_MS_ADPCM",
  "SOX_ENCODING_IMA_ADPCM",
  "SOX_ENCODING_OKI_ADPCM",
  "SOX_ENCODING_DPCM",
  "SOX_ENCODING_DWVW",
  "SOX_ENCODING_DWVWN",
  "SOX_ENCODING_GSM",
  "SOX_ENCODING_MP3",
  "SOX_ENCODING_VORBIS",
  "SOX_ENCODING_AMR_WB",
  "SOX_ENCODING_AMR_NB",
  "SOX_ENCODING_CVSD",
  "SOX_ENCODING_LPC10",
  "SOX_ENCODING_OPUS",
  "SOX_ENCODINGS"
}
"""


class SoxEffects(object):

    def __init__(self, normalization=True, channels_first=True, out_siginfo=None, out_encinfo=None, filetype="raw"):
        self.input_file = None
        self.chain = []
        self.MAX_EFFECT_OPTS = 20
        self.out_siginfo = out_siginfo
        self.out_encinfo = out_encinfo
        self.filetype = filetype
        self.normalization = normalization
        self.channels_first = channels_first

    def sox_check_effect(self, e):
        if e.lower() not in EFFECT_NAMES:
            raise LookupError("Effect name, {}, not valid".format(e.lower()))
        return e.lower()

    def sox_append_effect_to_chain(self, ename, eargs=None):
        e = torchaudio.SoxEffect()
        # check if we have a valid effect
        ename = self.sox_check_effect(ename)
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
        # initialize output tensor
        if out is not None:
            torchaudio.check_input(out)
        else:
            out = torch.FloatTensor()
        if not len(self.chain):
            e = torchaudio.SoxEffect()
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
        self.chain = []

    def set_input_file(self, input_file):
        self.input_file = input_file

    # https://stackoverflow.com/questions/12472338/flattening-a-list-recursively
    # convenience function to flatten list recursively
    def _flatten(self, x):
        if x == []:
            return []
        if isinstance(x[0], list):
            return self._flatten(x[:1]) + self._flatten(x[:1])
        return [str(a) for a in x[:1]] + self._flatten(x[1:])
