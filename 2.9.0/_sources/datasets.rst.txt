.. py:module:: torchaudio.datasets

torchaudio.datasets
====================

All datasets are subclasses of :class:`torch.utils.data.Dataset`
and have ``__getitem__`` and ``__len__`` methods implemented.

Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples parallelly using :mod:`torch.multiprocessing` workers.
For example:

.. code::

   yesno_data = torchaudio.datasets.YESNO('.', download=True)
   data_loader = torch.utils.data.DataLoader(
       yesno_data,
       batch_size=1,
       shuffle=True,
       num_workers=args.nThreads)

.. currentmodule:: torchaudio.datasets

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/dataset_class.rst

   CMUARCTIC
   CMUDict
   COMMONVOICE
   DR_VCTK
   FluentSpeechCommands
   GTZAN
   IEMOCAP
   LibriMix
   LIBRISPEECH
   LibriLightLimited
   LIBRITTS
   LJSPEECH
   MUSDB_HQ
   QUESST14
   Snips
   SPEECHCOMMANDS
   TEDLIUM
   VCTK_092
   VoxCeleb1Identification
   VoxCeleb1Verification
   YESNO
