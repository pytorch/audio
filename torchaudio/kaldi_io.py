# To use this file, the dependency (https://github.com/vesis84/kaldi-io-for-python)
# needs to be installed. This is a light wrapper around kaldi_io that returns
# torch.Tensors.
import numpy as np
import torch


__all__ = [
    'read_vec_int_ark',
    'read_vec_flt_scp',
    'read_vec_flt_ark',
    'read_mat_scp',
    'read_mat_ark',
]


def _default_not_imported_method():
    raise ImportError('Could not import kaldi_io. Did you install it?')


def _wrap_method(fn, convert_contiguous=False):
    # type: (Function, bool) -> Function
    """ Takes a method with the signature (file name/descriptor) -> generator(string, ndarray)
    and converts it to (file name/descriptor) -> generator(string, Tensor).
    convert_contiguous determines whether the array should be converted into a
    contiguous layout.
    """
    def _wrapped_fn(file_or_fd):
        for key, np_arr in fn(file_or_fd):
            if convert_contiguous:
                np_arr = np.ascontiguousarray(np_arr)
            yield key, torch.from_numpy(np_arr)
    return _wrapped_fn


#: Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
#:
#: file_or_fd : ark, gzipped ark, pipe or opened file descriptor.
#:
#: Example, read ark to a 'dictionary':
#:
#: >>> # generator(key,vec) = torchaudio.kaldi_io.read_vec_int_ark(file_or_fd)
#: >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_int_ark(file) }
read_vec_int_ark = _default_not_imported_method

#: Create generator of (key,vector<float32/float64>) tuples, read according to kaldi scp.
#:
#: file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
#:
#: Example, read scp to a 'dictionary':
#:
#: >>> # generator(key,vec) = torchaudio.kaldi_io.read_vec_flt_scp(file_or_fd)
#: >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_flt_scp(file) }
read_vec_flt_scp = _default_not_imported_method
#: Create generator of (key,vector<float32/float64>) tuples, which reads from the ark file/stream.
#:
#: file_or_fd : ark, gzipped ark, pipe or opened file descriptor.
#:
#: Example, read ark to a 'dictionary':
#:
#: >>> # generator(key,vec) = torchaudio.kaldi_io.read_vec_flt_ark(file_or_fd)
#: >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_flt_ark(file) }
read_vec_flt_ark = _default_not_imported_method

#: Create generator of (key,matrix<float32/float64>) tuples, read according to kaldi scp.
#:
#: file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
#:
#: Example, read scp to a 'dictionary':
#:
#: >>> # generator(key,mat) = torchaudio.kaldi_io.read_mat_scp(file_or_fd)
#: >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_scp(file) }
read_mat_scp = _default_not_imported_method
#: Create generator of (key,matrix<float32/float64>) tuples, which reads from the ark file/stream.
#:
#: file_or_fd : ark, gzipped ark, pipe or opened file descriptor.
#:
#: Example, read ark to a 'dictionary':
#:
#: >>> # generator(key,mat) = torchaudio.kaldi_io.read_mat_ark(file_or_fd)
#: >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_ark(file) }
read_mat_ark = _default_not_imported_method

try:
    import kaldi_io

    # Overwrite methods
    # Elements from int32 vector are sored in tuples: (sizeof(int32), value)
    # so strides are (5,) instead of (4,) which will throw an error in from_numpy
    # as it expects strides to be a multiple of 4 (int32).
    read_vec_int_ark = _wrap_method(kaldi_io.read_vec_int_ark, convert_contiguous=True)

    read_vec_flt_scp = _wrap_method(kaldi_io.read_vec_flt_scp)
    read_vec_flt_ark = _wrap_method(kaldi_io.read_vec_flt_ark)

    read_mat_scp = _wrap_method(kaldi_io.read_mat_scp)
    read_mat_ark = _wrap_method(kaldi_io.read_mat_ark)
except ImportError:
    pass
