def get_test_name(func, _, params):
    return f'{func.__name__}_{"_".join(str(p) for p in params.args)}'
