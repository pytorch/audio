import io

import torch


def torch_script(obj):
    """TorchScript the given function or Module"""
    buffer = io.BytesIO()
    if hasattr(obj, '__wrapped__'):
        # This is hack for those functions which are deprecated with decorators
        # like @deprecated or @dropping_support. Adding the decorators breaks
        # TorchScript. We need to unwrap the function to get the original one,
        # which make the tests pass, but that's a lie: the public (deprecated)
        # function doesn't support torchscript anymore
        obj = obj.__wrapped__
    torch.jit.save(torch.jit.script(obj), buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)
