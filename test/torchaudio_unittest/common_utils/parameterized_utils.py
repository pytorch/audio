import json
from itertools import product

from parameterized import param, parameterized

from .data_utils import get_asset_path


def load_params(*paths):
    with open(get_asset_path(*paths), 'r') as file:
        return [param(json.loads(line)) for line in file]


def nested_params(*params):
    def _name_func(func, _, params):
        strs = []
        for arg in params.args:
            if isinstance(arg, tuple):
                strs.append("_".join(str(a) for a in arg))
            else:
                strs.append(str(arg))
        return f'{func.__name__}_{"_".join(strs)}'

    return parameterized.expand(
        list(product(*params)),
        name_func=_name_func
    )
