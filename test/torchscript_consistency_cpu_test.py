from common_utils import define_test_suites
from torchscript_consistency_impl import Functional, Transforms


define_test_suites(globals(), [Functional, Transforms], devices=['cpu'])
