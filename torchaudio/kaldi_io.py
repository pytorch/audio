# To use this file, the dependency (https://github.com/vesis84/kaldi-io-for-python)
# needs to be installed. This is a light wrapper around kaldi_io that returns
# torch.Tensors.
from typing import Any, Callable, Iterable, Tuple

import torch
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils

if _mod_utils.is_module_available("numpy"):
    import numpy as np


__all__ = [
    "read_vec_int_ark",
    "read_vec_flt_scp",
    "read_vec_flt_ark",
    "read_mat_scp",
    "read_mat_ark",
]


def _convert_method_output_to_tensor(
    file_or_fd: Any, fn: Callable, convert_contiguous: bool = False
) -> Iterable[Tuple[str, Tensor]]:
    r"""Takes a method invokes it. The output is converted to a tensor.

    Args:
        file_or_fd (str/FileDescriptor): File name or file descriptor
        fn (Callable): Function that has the signature (file name/descriptor) and converts it to
            Iterable[Tuple[str, Tensor]].
        convert_contiguous (bool, optional): Determines whether the array should be converted into a
            contiguous layout. (Default: ``False``)

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is vec/mat
    """
    for key, np_arr in fn(file_or_fd):
        if convert_contiguous:
            np_arr = np.ascontiguousarray(np_arr)
        yield key, torch.from_numpy(np_arr)


@_mod_utils.requires_module("kaldi_io", "numpy")
def read_vec_int_ark(file_or_fd: Any) -> Iterable[Tuple[str, Tensor]]:
    r"""Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.

    Args:
        file_or_fd (str/FileDescriptor): ark, gzipped ark, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the vector read from file

    Example
        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_int_ark(file) }
    """

    import kaldi_io

    # Requires convert_contiguous to be True because elements from int32 vector are
    # sorted in tuples: (sizeof(int32), value) so strides are (5,) instead of (4,) which will throw an error
    # in from_numpy as it expects strides to be a multiple of 4 (int32).
    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_vec_int_ark, convert_contiguous=True)


@_mod_utils.requires_module("kaldi_io", "numpy")
def read_vec_flt_scp(file_or_fd: Any) -> Iterable[Tuple[str, Tensor]]:
    r"""Create generator of (key,vector<float32/float64>) tuples, read according to Kaldi scp.

    Args:
        file_or_fd (str/FileDescriptor): scp, gzipped scp, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the vector read from file

    Example
        >>> # read scp to a 'dictionary'
        >>> # d = { u:d for u,d in torchaudio.kaldi_io.read_vec_flt_scp(file) }
    """

    import kaldi_io

    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_vec_flt_scp)


@_mod_utils.requires_module("kaldi_io", "numpy")
def read_vec_flt_ark(file_or_fd: Any) -> Iterable[Tuple[str, Tensor]]:
    r"""Create generator of (key,vector<float32/float64>) tuples, which reads from the ark file/stream.

    Args:
        file_or_fd (str/FileDescriptor): ark, gzipped ark, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the vector read from file

    Example
        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_vec_flt_ark(file) }
    """

    import kaldi_io

    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_vec_flt_ark)


@_mod_utils.requires_module("kaldi_io", "numpy")
def read_mat_scp(file_or_fd: Any) -> Iterable[Tuple[str, Tensor]]:
    r"""Create generator of (key,matrix<float32/float64>) tuples, read according to Kaldi scp.

    Args:
        file_or_fd (str/FileDescriptor): scp, gzipped scp, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the matrix read from file

    Example
        >>> # read scp to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_scp(file) }
    """

    import kaldi_io

    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_mat_scp)


@_mod_utils.requires_module("kaldi_io", "numpy")
def read_mat_ark(file_or_fd: Any) -> Iterable[Tuple[str, Tensor]]:
    r"""Create generator of (key,matrix<float32/float64>) tuples, which reads from the ark file/stream.

    Args:
        file_or_fd (str/FileDescriptor): ark, gzipped ark, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the matrix read from file

    Example
        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_ark(file) }
    """

    import kaldi_io

    return _convert_method_output_to_tensor(file_or_fd, kaldi_io.read_mat_ark)
