.. role:: hidden
    :class: hidden-section

torchaudio.prototype.transducer
===============================

.. currentmodule:: torchaudio.prototype.transducer

.. note::

    The RNN transducer loss is a prototype feature, see `here <https://pytorch.org/audio>`_ to learn more about the nomenclature. It is only available within the nightlies, and also needs to be imported explicitly using: :code:`from torchaudio.prototype.transducer import rnnt_loss, RNNTLoss`.

rnnt_loss
---------

.. autofunction:: rnnt_loss

:hidden:`RNNTLoss`
~~~~~~~~~~~~~~~~~~

.. autoclass:: RNNTLoss

  .. automethod:: forward
