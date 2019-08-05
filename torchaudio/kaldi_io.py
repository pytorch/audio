from __future__ import absolute_import, division, print_function, unicode_literals
# To use this file, the dependency (https://github.com/vesis84/kaldi-io-for-python)
# needs to be installed. This is a light wrapper around kaldi_io that returns
# torch.Tensors.
import torch
from torchaudio.common_utils import IMPORT_KALDI_IO, IMPORT_NUMPY

if IMPORT_NUMPY:
    import numpy as np

if IMPORT_KALDI_IO:
    import kaldi_io


__all__ = [
    'read_vec_int_ark',
    'read_vec_flt_scp',
    'read_vec_flt_ark',
    'read_mat_scp',
    'read_mat_ark',
]


def _convert_method_output_to_tensor(file_or_fd, fn, convert_contiguous=False):
    r""" Takes a method invokes it. The output is converted to a tensor.

    Arguments:
        file_or_fd (string/File Descriptor): file name or file descriptor.
        fn (Function): function that has the signature (file name/descriptor) -> generator(string, ndarray)
            and converts it to (file name/descriptor) -> generator(string, Tensor).
        convert_contiguous (bool): determines whether the array should be converted into a
            contiguous layout.

    Returns:
        generator[key (string), vec/mat (Tensor)]
    """
    if not IMPORT_KALDI_IO:
        raise ImportError('Could not import kaldi_io. Did you install it?')

    for key, np_arr in fn(file_or_fd):
        if convert_contiguous:
            np_arr = np.ascontiguousarray(np_arr)
        yield key, torch.from_numpy(np_arr)


def read_vec_int_ark(file_or_fd):
    r"""Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.

    Arguments:
        file_or_fd (string/File Descriptor): ark, gzipped ark, pipe or opened file descriptor.

    Returns:
        generator[key (string), vec (Tensor)]

    Example::

        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_int_ark(file) }
    """
    # Requires convert_contiguous to be True because elements from int32 vector are
    # sored in tuples: (sizeof(int32), value) so strides are (5,) instead of (4,) which will throw an error
    # in from_numpy as it expects strides to be a multiple of 4 (int32).
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_vec_int_ark, convert_contiguous=True)


def read_vec_flt_scp(file_or_fd):
    r"""Create generator of (key,vector<float32/float64>) tuples, read according to kaldi scp.

    Arguments:
        file_or_fd (string/File Descriptor): scp, gzipped scp, pipe or opened file descriptor.

    Returns:
        generator[key (string), vec (Tensor)]

    Example::

        >>> # read scp to a 'dictionary'
        >>> # d = { u:d for u,d in torchaudio.kaldi_io.read_vec_flt_scp(file) }
    """
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_vec_flt_scp)


def read_vec_flt_ark(file_or_fd):
    r"""Create generator of (key,vector<float32/float64>) tuples, which reads from the ark file/stream.

    Arguments:
        file_or_fd (string/File Descriptor): ark, gzipped ark, pipe or opened file descriptor.

    Returns:
        generator[key (string), vec (Tensor)]

    Example::

        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_flt_ark(file) }
    """
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_vec_flt_ark)


def read_mat_scp(file_or_fd):
    r"""Create generator of (key,matrix<float32/float64>) tuples, read according to kaldi scp.

    Arguments:
        file_or_fd (string/File Descriptor): scp, gzipped scp, pipe or opened file descriptor.

    Returns:
        generator[key (string), mat (Tensor)]

    Example::

        >>> # read scp to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_scp(file) }
    """
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_mat_scp)


def read_mat_ark(file_or_fd):
    r"""Create generator of (key,matrix<float32/float64>) tuples, which reads from the ark file/stream.

    Arguments:
        file_or_fd (string/File Descriptor): ark, gzipped ark, pipe or opened file descriptor.

    Returns:
        generator[key (string), mat (Tensor)]

    Example::

        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_ark(file) }
    """
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_mat_ark)
