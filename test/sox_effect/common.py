def name_func(func, _, params):
    if isinstance(params.args[0], str):
        args = "_".join([str(arg) for arg in params.args])
    else:
        args = "_".join([str(arg) for arg in params.args[0]])
    return f'{func.__name__}_{args}'
