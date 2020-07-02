def name_func(func, _, params):
    return f'{func.__name__}_{"_".join(str(arg) for arg in params.args)}'
