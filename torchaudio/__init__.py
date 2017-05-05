import torch

from cffi import FFI
ffi = FFI()
from ._ext import th_sox

def load(filename, out=None):
    if out is not None:
        assert torch.is_tensor(out)
        assert not out.is_cuda
    else:
        out = torch.FloatTensor()

    if isinstance(out, torch.FloatTensor):
        func = th_sox.libthsox_Float_read_audio_file
    elif isinstance(out, torch.DoubleTensor):
        func = th_sox.libthsox_Double_read_audio_file
    elif isinstance(out, torch.ByteTensor):
        func = th_sox.libthsox_Byte_read_audio_file
    elif isinstance(out, torch.CharTensor):
        func = th_sox.libthsox_Char_read_audio_file
    elif isinstance(out, torch.ShortTensor):
        func = th_sox.libthsox_Short_read_audio_file
    elif isinstance(out, torch.IntTensor):
        func = th_sox.libthsox_Int_read_audio_file
    elif isinstance(out, torch.LongTensor):
        func = th_sox.libthsox_Long_read_audio_file
        
    sample_rate_p = ffi.new('int*')    
    func(bytes(filename), out, sample_rate_p)
    sample_rate = sample_rate_p[0]
    return out, sample_rate
