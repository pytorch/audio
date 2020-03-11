import torch
from torch import nn

__all__ = ["Wav2Letter", "wav2letter"]


class Wav2Letter(nn.Module):
    r"""Wav2Letter model architecture from the `"Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"
     <https://arxiv.org/abs/1609.03193>`_ paper.

    Args:
        num_classes (int, optional): Number of classes to be classified. (Default: ``40``)
        version (str, optional): Wav2Letter can use as input: ``waveform``, ```power_spectrum``
         or ```mfcc``. (Default: ``waveform``)
        n_input_features (int, optional): If is used ``power_spectrum`` or ```mfcc`` must be
        specified the number of features that were extracted. (Default: ``None``)
    """

    def __init__(self, num_classes=40, version="waveform", n_input_features=None):
        super(Wav2Letter, self).__init__()
        n_input_features = 250 if not n_input_features else n_input_features

        acoustic_model = nn.Sequential(
            nn.Conv1d(in_channels=n_input_features, out_channels=250, kernel_size=48, stride=2, padding=23),
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
            nn.Conv1d(in_channels=250, out_channels=2000, kernel_size=32, stride=1, padding=15),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=2000, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=2000, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        if version == "waveform":
            waveform_model = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=250, kernel_size=250, stride=160, padding=45),
                nn.ReLU(inplace=True)
            )
            self.acoustic_model = nn.Sequential(waveform_model, acoustic_model)

        if version in ["power_spectrum", "mfcc"]:
            self.acoustic_model = acoustic_model

    def forward(self, x):
        # type: (Tensor) -> Tensor
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, n_features, input_length).

        Returns:
            torch.Tensor: Predictor tensor of dimension (input_length, batch_size, number_of_classes).
        """

        x = self.acoustic_model(x)
        x = nn.functional.log_softmax(x, dim=1)
        x = x.permute(2, 0, 1)
        return x


def wav2letter(**kwargs):
    r"""Wav2Letter model architecture from the `"Wav2Letter: an End-to-End ConvNet-based Speech Recognition System"
     <https://arxiv.org/abs/1609.03193>`_ paper.
    """
    model = Wav2Letter(**kwargs)
    return model
