import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['torchaudio/src/th_sox.c']
headers = [
    'torchaudio/src/th_sox.h',
]
defines = []

ffi = create_extension(
    'torchaudio._ext.th_sox',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    libraries=['sox'],
    include_dirs=['torchaudio/src'],
)

if __name__ == '__main__':
    ffi.build()
