torchaudio.datasets
====================

All datasets are subclasses of :class:`torch.utils.data.Dataset`
and have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using ``torch.multiprocessing`` workers.
For example: ::

    yesno_data = torchaudio.datasets.YESNO('.', download=True)
    data_loader = torch.utils.data.DataLoader(yesno_data,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=args.nThreads)

The following datasets are available:

.. contents:: Datasets
    :local:

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.


.. currentmodule:: torchaudio.datasets


CMUARCTIC
~~~~~~~~~

.. autoclass:: CMUARCTIC
  :members:
  :special-members: __getitem__


COMMONVOICE
~~~~~~~~~~~

.. autoclass:: COMMONVOICE
  :members:
  :special-members: __getitem__


GTZAN
~~~~~

.. autoclass:: GTZAN
  :members:
  :special-members: __getitem__


LIBRISPEECH
~~~~~~~~~~~

.. autoclass:: LIBRISPEECH
  :members:
  :special-members: __getitem__


LIBRITTS
~~~~~~~~

.. autoclass:: LIBRITTS
  :members:
  :special-members: __getitem__


LJSPEECH
~~~~~~~~

.. autoclass:: LJSPEECH
  :members:
  :special-members: __getitem__


SPEECHCOMMANDS
~~~~~~~~~~~~~~

.. autoclass:: SPEECHCOMMANDS
  :members:
  :special-members: __getitem__


TEDLIUM
~~~~~~~~~~~~~~

.. autoclass:: TEDLIUM
  :members:
  :special-members: __getitem__


VCTK
~~~~

.. autoclass:: VCTK
  :members:
  :special-members: __getitem__


VCTK_092
~~~~~~~~

.. autoclass:: VCTK_092
  :members:
  :special-members: __getitem__


YESNO
~~~~~

.. autoclass:: YESNO
  :members:
  :special-members: __getitem__
