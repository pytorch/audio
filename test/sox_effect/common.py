import json

from parameterized import param

from ..common_utils import get_asset_path


def name_func(func, _, params):
    if isinstance(params.args[0], str):
        args = "_".join([str(arg) for arg in params.args])
    else:
        args = "_".join([str(arg) for arg in params.args[0]])
    return f'{func.__name__}_{args}'


def load_params(*paths):
    params = []
    with open(get_asset_path(*paths), 'r') as file:
        for line in file:
            data = json.loads(line)
            for effect in data['effects']:
                for i, arg in enumerate(effect):
                    if arg.startswith("<ASSET_DIR>"):
                        effect[i] = arg.replace("<ASSET_DIR>", get_asset_path())
            params.append(param(data))
    return params
