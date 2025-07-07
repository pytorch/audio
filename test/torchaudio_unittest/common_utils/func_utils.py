import io

import torch


def torch_script(obj):
    """TorchScript the given function or Module"""
    buffer = io.BytesIO()
    if hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__
    torch.jit.save(torch.jit.script(obj), buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)
