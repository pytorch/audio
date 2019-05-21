# To use this file, the dependency (https://github.com/vesis84/kaldi-io-for-python)
# needs to be installed. This is a light wrapper around kaldi_io that returns
# torch.Tensors.
import numpy as np
import torch


def default_not_imported_method():
    raise ImportError('Could not import kaldi_io')


def wrap_method(fn, convert_contiguous=False):
    # type: (Function, bool) -> Function
    """ Takes a method with the signature (file name/descriptor) -> generator(string, ndarray)
    and converts it to (file name/descriptor) -> generator(string, Tensor).
    convert_contiguous determines whether the array should be converted into a
    contiguous layout.
    """
    def wrapped_fn(file_or_fd):
        for key, np_arr in fn(file_or_fd):
            if convert_contiguous:
                np_arr = np.ascontiguousarray(np_arr)
            yield key, torch.from_numpy(np_arr)
    return wrapped_fn


read_vec_int_ark = default_not_imported_method

read_vec_flt_scp = default_not_imported_method
read_vec_flt_ark = default_not_imported_method

read_mat_scp = default_not_imported_method
read_mat_ark = default_not_imported_method

try:
    import kaldi_io

    # Overwrite methods
    # Elements from int32 vector are sored in tuples: (sizeof(int32), value)
    # so strides are (5,) instead of (4,) which will throw an error in from_numpy
    # as it expects strides to be a multiple of 4 (int32).
    read_vec_int_ark = wrap_method(kaldi_io.read_vec_int_ark, convert_contiguous=True)

    read_vec_flt_scp = wrap_method(kaldi_io.read_vec_flt_scp)
    read_vec_flt_ark = wrap_method(kaldi_io.read_vec_flt_ark)

    read_mat_scp = wrap_method(kaldi_io.read_mat_scp)
    read_mat_ark = wrap_method(kaldi_io.read_mat_ark)
except ImportError:
    pass
