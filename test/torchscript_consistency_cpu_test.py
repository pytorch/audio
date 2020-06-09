from parameterized import parameterized_class

from .common_utils import TestCase, common_test_class_parameters
from .torchscript_consistency_impl import Functional, Transforms

parameters = list(common_test_class_parameters(devices=['cpu']))
@parameterized_class(parameters)
class TestFunctional(Functional, TestCase):
    pass


@parameterized_class(parameters)
class TestTransforms(Transforms, TestCase):
    pass
