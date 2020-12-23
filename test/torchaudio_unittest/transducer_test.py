import torch

from torchaudio_unittest import common_utils
from torchaudio.prototype.transducer import RNNTLoss


class TransducerTester:
    def test_basic_backward(self):
        rnnt_loss = RNNTLoss()

        acts = torch.FloatTensor(
            [
                [
                    [
                        [0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.1, 0.1],
                        [0.1, 0.1, 0.2, 0.8, 0.1],
                    ],
                    [
                        [0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.2, 0.1, 0.1],
                        [0.7, 0.1, 0.2, 0.1, 0.1],
                    ],
                ]
            ]
        )
        labels = torch.IntTensor([[1, 2]])
        act_length = torch.IntTensor([2])
        label_length = torch.IntTensor([2])

        acts = acts.to(self.device)
        labels = labels.to(self.device)
        act_length = act_length.to(self.device)
        label_length = label_length.to(self.device)

        acts = torch.autograd.Variable(acts, requires_grad=True)
        labels = torch.autograd.Variable(labels)
        act_length = torch.autograd.Variable(act_length)
        label_length = torch.autograd.Variable(label_length)

        loss = rnnt_loss(acts, labels, act_length, label_length)
        loss.backward()


@common_utils.skipIfNoTransducer
class CPUTransducerTester(TransducerTester, common_utils.PytorchTestCase):
    device = "cpu"


@common_utils.skipIfNoCudaTransducer
class GPUTransducerTester(TransducerTester, common_utils.PytorchTestCase):
    device = "cuda"
