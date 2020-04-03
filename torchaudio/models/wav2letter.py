from typing import Optional

from torch import Tensor
from torch import nn

__all__ = ["Wav2Letter"]


class Wav2Letter(nn.Module):
    r"""Wav2Letter model architecture from the `"Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"
     <https://arxiv.org/abs/1609.03193>`_ paper.

     :math:`\text{padding} = \frac{\text{ceil}(\text{kernel} - \text{stride})}{2}`

    Args:
        num_classes (int, optional): Number of classes to be classified. (Default: ``40``)
        version (str, optional): Wav2Letter can use as input: ``waveform``, ``power_spectrum``
         or ``mfcc`` (Default: ``waveform``).
        num_features (int, optional): Number of input features that the network will receive (Default: ``1``).
    """

    def __init__(self, num_classes: int = 40,
                 version: str = "waveform",
                 num_features: Optional[int] = 1) -> None:
        super(Wav2Letter, self).__init__()

        acoustic_model = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=250, kernel_size=48, stride=2, padding=23),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=250, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        if version == "waveform":
            waveform_model = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=250, kernel_size=250, stride=160, padding=45),
                nn.ReLU(inplace=True)
            )
            self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)

        if version in ["power_spectrum", "mfcc"]:
            self.acoustic_model = acoustic_model

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): Tensor of dimension (batch_size, n_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (input_length, batch_size, number_of_classes).
        """

        x = self.acoustic_model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
