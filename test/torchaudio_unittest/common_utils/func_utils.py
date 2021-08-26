import io
import torch


def torch_script(obj):
    """TorchScript the given function or Module"""
    buffer_ = io.BytesIO()
    torch.jit.save(torch.jit.script(obj), buffer_)
    buffer_.seek(0)
    return torch.jit.load(buffer_)
