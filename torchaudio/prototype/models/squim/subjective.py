from typing import Tuple

import torch
import torch.nn as nn
import torchaudio


class AttPool(nn.Module):
    """Attention-Pooling module that estimates the attention score.

    Args:
        input_dim (int): Input feature dimension.
        att_dim (int): Attention Tensor dimension.
    """

    def __init__(self, input_dim: int, att_dim: int):
        super(AttPool, self).__init__()

        self.linear1 = nn.Linear(input_dim, 1)
        self.linear2 = nn.Linear(input_dim, att_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and pooling.

        Args:
            x (torch.Tensor): Input Tensor with dimensions `(batch, time, feature_dim)`.

        Returns:
            (torch.Tensor): Attention score with dimensions `(batch, att_dim)`.
        """

        att = self.linear1(x)  # (batch, time, 1)
        att = att.transpose(2, 1)  # (batch, 1, time)
        att = nn.functional.softmax(att, dim=2)
        x = torch.matmul(att, x).squeeze(1)  # (batch, input_dim)
        x = self.linear2(x)  # (batch, att_dim)
        return x


class Predictor(nn.Module):
    """Prediction module that apply pooling and attention, then predict subjective metric scores.

    Args:
        input_dim (int): Input feature dimension.
        att_dim (int): Attention Tensor dimension.
    """

    def __init__(self, input_dim: int, att_dim: int):
        super(Predictor, self).__init__()
        self.att_pool_layer = AttPool(input_dim, att_dim)
        self.att_dim = att_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict subjective evaluation metric score.

        Args:
            x (torch.Tensor): Input Tensor with dimensions `(batch, time, feature_dim)`.

        Returns:
            (torch.Tensor): Subjective metric score. Tensor with dimensions `(batch,)`.
        """
        x = self.att_pool_layer(x)
        x = nn.functional.softmax(x, dim=1)
        B = torch.linspace(0, 4, steps=self.att_dim, device=x.device)
        x = (x * B).sum(dim=1)
        return x


class SquimSubjective(nn.Module):
    """Speech Quality and Intelligibility Measures (SQUIM) model that predicts **subjective** metric scores
    for speech enhancement (e.g., Mean Opinion Score (MOS)). The model is adopted from *NORESQA-MOS*
    :cite:`manocha2022speech` which predicts MOS scores given the input speech and a non-matching reference.

    Args:
        ssl_model (torch.nn.Module): The self-supervised learning model for feature extraction.
        projector (torch.nn.Module): Projection layer that projects SSL feature to a lower dimension.
        predictor (torch.nn.Module): Predict the subjective scores.
    """

    def __init__(self, ssl_model: nn.Module, projector: nn.Module, predictor: nn.Module):
        super(SquimSubjective, self).__init__()
        self.ssl_model = ssl_model
        self.projector = projector
        self.predictor = predictor

    def _align_shapes(self, waveform: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cut or pad the reference Tensor to make it aligned with waveform Tensor.

        Args:
            waveform (torch.Tensor): Input waveform for evaluation. Tensor with dimensions `(batch, time)`.
            reference (torch.Tensor): Non-matching clean reference. Tensor with dimensions `(batch, time_ref)`.

        Returns:
            (torch.Tensor, torch.Tensor): The aligned waveform and reference Tensors
                with same dimensions `(batch, time)`.
        """
        T_waveform = waveform.shape[-1]
        T_reference = reference.shape[-1]
        if T_reference < T_waveform:
            num_padding = T_waveform // T_reference + 1
            reference = torch.cat([reference for _ in range(num_padding)], dim=1)
        return waveform, reference[:, :T_waveform]

    def forward(self, waveform: torch.Tensor, reference: torch.Tensor):
        """Predict subjective evaluation metric score.

        Args:
            waveform (torch.Tensor): Input waveform for evaluation. Tensor with dimensions `(batch, time)`.
            reference (torch.Tensor): Non-matching clean reference. Tensor with dimensions `(batch, time_ref)`.

        Returns:
            (torch.Tensor): Subjective metric score. Tensor with dimensions `(batch,)`.
        """
        waveform, reference = self._align_shapes(waveform, reference)
        waveform = self.projector(self.ssl_model.extract_features(waveform)[0][-1])
        reference = self.projector(self.ssl_model.extract_features(reference)[0][-1])
        concat = torch.cat((reference, waveform), dim=2)
        score_diff = self.predictor(concat)  # Score difference compared to the reference
        return 5 - score_diff


def squim_subjective_model(
    ssl_type: str,
    feat_dim: int,
    proj_dim: int,
    att_dim: int,
) -> SquimSubjective:
    """Build a custome :class:`torchaudio.prototype.models.SquimSubjective` model.

    Args:
        ssl_type (str): Type of self-supervised learning (SSL) models.
            Must be one of ["wav2vec2_base", "wav2vec2_large"].
        feat_dim (int): Feature dimension of the SSL feature representation.
        proj_dim (int): Output dimension of projection layer.
        att_dim (int): Dimension of attention scores.
    """
    ssl_model = getattr(torchaudio.models, ssl_type)()
    projector = nn.Linear(feat_dim, proj_dim)
    predictor = Predictor(proj_dim * 2, att_dim)
    return SquimSubjective(ssl_model, projector, predictor)


def squim_subjective_base() -> SquimSubjective:
    """Build :class:`torchaudio.prototype.models.SquimSubjective` model with default arguments."""
    return squim_subjective_model(
        ssl_type="wav2vec2_base",
        feat_dim=768,
        proj_dim=32,
        att_dim=5,
    )
