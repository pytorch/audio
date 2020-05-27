from . import common_utils
from .kaldi_compatibility_impl import Kaldi


common_utils.define_test_suites(globals(), [Kaldi], devices=['cuda'])
