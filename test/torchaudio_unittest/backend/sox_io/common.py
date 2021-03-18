def name_func(func, _, params):
    return f'{func.__name__}_{"_".join(str(arg) for arg in params.args)}'


def get_enc_params(dtype):
    if dtype == 'float32':
        return 'PCM_F', 32
    if dtype == 'int32':
        return 'PCM_S', 32
    if dtype == 'int16':
        return 'PCM_S', 16
    if dtype == 'uint8':
        return 'PCM_U', 8
    raise ValueError(f'Unexpected dtype: {dtype}')
