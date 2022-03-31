Supported Features
==================

Each TorchAudio API supports a subset of PyTorch features, such as
devices and data types.
Supported features are indicated in API references like the following:

.. dtypes:: fp32 fp64

.. devices:: CPU CUDA

.. properties:: Autograd TorchScript

These icons mean that they are verified through automated testing.

.. note::

   Missing feature icons mean that they are not tested, and this can mean
   different things, depending on the API.

   1. The API is compatible with the feature but not tested.
   2. The API is not compatible with the feature.

   In case of 2, the API might explicitly raise an error, but that is not guaranteed.
   For example, APIs without an Autograd badge might throw an error during backpropagation,
   or silently return a wrong gradient.

If you use an API that hasn't been labeled as supporting a feature, you might want to first verify that the
feature works fine.

Data Types
----------

Data type badges indicate whether APIs for numerical computations are tested on the specific input data type.

Floating Point Data Types
^^^^^^^^^^^^^^^^^^^^^^^^^

.. dtypes:: fp32 fp64

Floating point types represents real-valued quantities, including raw waveform, power spectrogram and other processed features.
  
Complex Data Types
^^^^^^^^^^^^^^^^^^

.. dtypes:: complex64 complex128

Complex data types are mainly used to represent the quantities in frequency domain. Most often, these are raw output from Fourier Transform.

.. note::

   In most cases, input data type and output data type are identical, but APIs which use Fourier transforms take float point data types and return corresponding complex data types. Similarly, APIs which use inverse Fourier transforms or estimate approximation take complex data types and return corresponding floating point data types. For example :py:func:`torchaudio.transforms.Spectrogram` returns ``torch.complex64`` when the input is ``torch.float32`` and :py:func:`torchaudio.transforms.InverseSpectrogram` returns ``torch.float32`` when the input is ``torch.complex64``.

.. note::

   APIs which take multiple input Tensors mostly expect Tensors to be the same data types. An exception to this is when a Tensor represents length or label. Unless noted otherwise, APIs expect length or lanel Tensors to be ``torch.float32`` regardless of the data type of feature Tensors.

Devices
-------

CPU
^^^

.. devices:: CPU

TorchAudio APIs that support CPU can perform their computation on CPU tensors.


CUDA
^^^^

.. devices:: CUDA

TorchAudio APIs that support CUDA can perform their computation on CUDA devices.

In case of functions, move the tensor arguments to CUDA device before passing them to a function.

For example:

.. code:: python

   cuda = torch.device("cuda")
          
   waveform = waveform.to(cuda)
   spectrogram = torchaudio.functional.spectrogram(waveform)

Classes with CUDA support are implemented with :py:func:`torch.nn.Module`.
It is also necessary to move the instance to CUDA device, before passing CUDA tensors.

For example:

.. code:: python

   cuda = torch.device("cuda")

   resampler = torchaudio.transforms.Resample(8000, 16000)
   resampler.to(cuda)

   waveform.to(cuda)
   resampled = resampler(waveform)


Properties
----------

Autograd
^^^^^^^^

.. properties:: Autograd

TorchAudio APIs with autograd support can correctly backpropagate gradients.

For the basics of autograd, please refer to this `tutorial <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_.

.. note::

   APIs without this mark may or may not raise an error during backpropagation.
   The absence of an error raised during backpropagation does not necessarily mean the gradient is correct.

TorchScript
^^^^^^^^^^^

.. properties:: TorchScript

TorchAudio APIs with TorchScript support can be serialized and executed in non-Python environments.

For details on TorchScript, please refer to the `documentation <https://pytorch.org/docs/stable/jit.html>`_.
