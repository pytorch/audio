Supported Features
==================

Each TorchAudio API supports different set of PyTorch features, such as different devices and data types.
Supported features are indicated in API references like the following.

.. devices:: CPU CUDA

.. features:: Autograd TorchScript

These icons mean that they are verified through automated testing.

.. note::

   Missing feature icons mean that they are not tested, and this can mean
   different things, depending on the API.

   1. The API is compatible with the feature but not tested.
   2. The API is not compatible with the feature.

   In case of 2, API might explicitly raise an error, but that is not always the case.
   For example, APIs with missing Autograd badge might throw an error on backward path,
   or silently return a wrong gradient.

If you use an API with the feature not supported, you might want to first verify that the
feature works fine.

Devices
-------

CPU
^^^

.. devices:: CPU

TorchAudio APIs that support CPU can perform the computation on CPU tensors.


CUDA
^^^^

.. devices:: CUDA

TorchAudio APIs that support CUDA can perform the computation on CUDA devices.

In case of functions, move the tensor arguments to CUDA device before passing them to a function.

For example

.. code:: python

   cuda = torch.device("cuda")
          
   waveform = waveform.to(cuda)
   spectrogram = torchaudio.functional.spectrogram(waveform)

The classes with CUDA support are implemented with :py:func:`torch.nn.Module`.
It is also necessary to move the instance to CUDA device, before passing CUDA tensors.

For example

.. code:: python

   cuda = torch.device("cuda")

   resampler = torchaudio.transforms.Resample(8000, 16000)
   resampler.to(cuda)

   waveform.to(cuda)
   resampled = resampler(waveform)


Features
--------

Autograd
^^^^^^^^

.. features:: Autograd

TorchAudio APIs with autograd support can correctly propagate the gradient in its backward path.

For the basics of autograd, please checkout this `tutorial <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_.

.. note::

   APIs without this mark may or may not raise and error in back propagation.
   The lack of error in back propagatoin does not mean the feature computes the gradient correctly.

TorchScript
^^^^^^^^^^^

.. features:: TorchScript

TorchAudio APIs with TorchScript support can be serialized and executed on non-Python environments.

For the detail of TorchScript, please checkout the `documentation <https://pytorch.org/docs/stable/jit.html>`_.
