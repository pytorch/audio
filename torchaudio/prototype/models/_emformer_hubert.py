from typing import List, Optional, Tuple

import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction


class FeatureEncoder(torch.nn.Module):
    """Extract features from log-mel spectrogram input. Consists of linear layer and time reduction layer.

    Args:
        input_dim (int): The feature dimension of log-mel spectrogram feature.
        output_dim (int): The feature dimension after linear layer.
        use_bias (bool): If ``True``, enable bias parameter in the linear layer.
        stride (int): Number of frames to merge for the output frame.
    """

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool, stride: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=use_bias)
        self.time_reduction = _TimeReduction(stride)

    def forward(
        self, input: torch.Tensor, lengths: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): The log-mel spectrogram input.
                Tensor with dimensions `(batch, time, input_dim)`.
            lengths (torch.Tensor or None): Valid length of each input sample.
                Tensor with dimension `(batch, )`.

        Returns:
            (torch.Tensor, torch.Tensor or None):
                torch.Tensor
                    Returned feature Tensor after linear layer and time reduction layer.
                    Tensor with dimensions `(batch, time // stride, output_dim)`.
                torch.Tensor or None
                    The reduced lengths Tensor.
        """
        output = self.linear(input)
        if lengths is None:
            B, T, _ = input.shape
            dummy_lengths = torch.full((B,), T)
            output, _ = self.time_reduction(output, dummy_lengths)
        else:
            output, lengths = self.time_reduction(output, lengths)
        return output, lengths


class EmformerEncoder(torch.nn.Module):
    """Emformer Encoder class for HuBERT pre-training. Consists of emformer module,
        linear layer and layer normalization layer.

    Args:
        emformer (torch.nn.Module):
            :py:class:`torchaudio.models.Emformer` module that consists of a list of emformer layers.
        output_linear (torch.nn.Module):
            Linear layer after emformer module.
        layer_norm (torch.nn.Module):
            Apply layer normalization to the output.
    """

    def __init__(
        self,
        emformer: torch.nn.Module,
        output_linear: torch.nn.Module,
        layer_norm: torch.nn.Module,
    ):
        super().__init__()
        self.emformer = emformer
        self.output_linear = output_linear
        self.layer_norm = layer_norm

    def forward(
        self,
        input: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): The input feature for emformer encoder.
                Tensor with dimensions `(batch, time, feature_dim)`.
            lengths (torch.Tensor or None): Valid length of each input sample.
                Tensor with dimension `(batch, )`.

        Returns:
            torch.Tensor: The feature Tensor after emformer encoder.
        """
        if lengths is None:
            B, T, _ = input.shape
            dummy_lengths = torch.full((B,), T)
            output, _ = self.emformer(input, dummy_lengths)
        else:
            output, lengths = self.emformer(input, lengths)
        output = self.output_linear(output)
        output = self.layer_norm(output)
        return output

    def extract_features(
        self,
        input: torch.Tensor,
        lengths: Optional[torch.Tensor],
        num_layers: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Extract output Tensors of the emformer layers.

        Args:
            input (torch.Tensor): The input feature for emformer encoder.
                Tensor with dimensions `(batch, time, feature_dim)`.
            lengths (torch.Tensor or None): Valid length of each input sample.
                Tensor with dimension `(batch, )`.
            num_layers (int or None, optional): If not ``None``, returns the first
                `num_layers` layers of Tensors as the output, otherwise returns the
                Tensors from all emformer layers.

        Returns:
            List[torch.Tensor]:
                Output Tensors of selected emformer layers.
        """
        if num_layers is not None:
            if not 0 < num_layers <= len(self.emformer.emformer_layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.emformer.emformer_layers)}]")

        ret: List[torch.Tensor] = []

        input = input.permute(1, 0, 2)
        right_context = self.emformer._gen_right_context(input)
        utterance = input[: input.size(0) - self.emformer.right_context_length]
        attention_mask = self.emformer._gen_attention_mask(utterance)
        mems = (
            self.emformer.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:-1]
            if self.emformer.use_mem
            else torch.empty(0).to(dtype=input.dtype, device=input.device)
        )
        output = utterance
        if lengths is None:
            B, T, _ = input.shape
            lengths = torch.full((B,), T)
        for layer in self.emformer.emformer_layers:
            output, right_context, mems = layer(output, lengths, right_context, mems, attention_mask)
            ret.append(output.permute(1, 0, 2))
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


def _get_emformer_feature_extractor(input_dim: int, output_dim: int, use_bias: bool, stride: int) -> FeatureEncoder:
    """Construct FeatureEncoder for emformer model.

    Args:
        input_dim (int): The feature dimension of log-mel spectrogram feature.
        output_dim (int): The feature dimension after linear layer.
        use_bias (bool): If ``True``, enable bias parameter in the linear layer.
        stride (int): Number of frames to merge for the output frame.

    Returns:
        FeatureEncoder: The resulting FeatureEncoder module.
    """
    return FeatureEncoder(input_dim, output_dim, use_bias, stride)


def _get_emformer_encoder(
    input_dim: int,
    output_dim: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
    segment_length: int,
    left_context_length: int,
    right_context_length: int,
    dropout: float,
    activation: str,
    max_memory_size: int,
    weight_init_scale_strategy: Optional[str],
    tanh_on_mem: bool,
) -> EmformerEncoder:
    """Construct EmformerEncoder for emformer model.

    Args:
        input_dim (int): The feature dimension of input Tensor.
        output_dim (int): The feature dimension after EmformerEncoder.
        num_heads (int): Number of attention heads in each Emformer layer.
        ffn_dim: (int): Hidden layer dimension of feedforward network.
        num_layers (int): Number of Emformer layers to instantiate.
        segment_length (int): Length of each input segment.
        left_context_length (int): Length of left context.
        right_context_length (int): Length of right context.
        dropout (float): Dropout probability.
        activation (str): Activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu").
        max_memory_size (int): Maximum number of memory elements to use.
        weight_init_scale_strategy (str or None): Per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``).
        tanh_on_mem (bool): If ``True``, applies tanh to memory elements.

    Returns:
        EmformerEncoder: The resulting EmformerEncoder module.
    """
    emformer = Emformer(
        input_dim=input_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        segment_length=segment_length,
        left_context_length=left_context_length,
        right_context_length=right_context_length,
        dropout=dropout,
        activation=activation,
        max_memory_size=max_memory_size,
        weight_init_scale_strategy=weight_init_scale_strategy,
        tanh_on_mem=tanh_on_mem,
    )
    output_linear = torch.nn.Linear(input_dim, output_dim)
    layer_norm = torch.nn.LayerNorm(output_dim)
    return EmformerEncoder(emformer, output_linear, layer_norm)


def emformer_hubert_model(
    extractor_input_dim: int,
    extractor_output_dim: int,
    extractor_use_bias: bool,
    extractor_stride: int,
    encoder_input_dim: int,
    encoder_output_dim: int,
    encoder_num_heads: int,
    encoder_ffn_dim: int,
    encoder_num_layers: int,
    encoder_segment_length: int,
    encoder_left_context_length: int,
    encoder_right_context_length: int,
    encoder_dropout: float,
    encoder_activation: str,
    encoder_max_memory_size: int,
    encoder_weight_init_scale_strategy: Optional[str],
    encoder_tanh_on_mem: bool,
    aux_num_out: Optional[int],
) -> Wav2Vec2Model:
    """Build a custom Emformer HuBERT model.

    Args:
        extractor_input_dim (int): The input dimension for feature extractor.
        extractor_output_dim (int): The output dimension after feature extractor.
        extractor_use_bias (bool): If ``True``, enable bias parameter in the linear layer of feature extractor.
        extractor_stride (int): Number of frames to merge for the output frame in feature extractor.
        encoder_input_dim (int): The input dimension for Emformer layer.
        encoder_output_dim (int): The output dimension after EmformerEncoder.
        encoder_num_heads (int): Number of attention heads in each Emformer layer.
        encoder_ffn_dim (int): Hidden layer dimension of feedforward network in Emformer.
        encoder_num_layers (int): Number of Emformer layers to instantiate.
        encoder_segment_length (int): Length of each input segment.
        encoder_left_context_length (int): Length of left context.
        encoder_right_context_length (int): Length of right context.
        encoder_dropout (float): Dropout probability.
        encoder_activation (str): Activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu").
        encoder_max_memory_size (int): Maximum number of memory elements to use.
        encoder_weight_init_scale_strategy (str or None): Per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``).
        encoder_tanh_on_mem (bool): If ``True``, applies tanh to memory elements.
        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting :py:class:`torchaudio.models.Wav2Vec2Model` model
            with a :py:class:`torchaudio.models.Emformer` encoder.
    """
    feature_extractor = _get_emformer_feature_extractor(
        extractor_input_dim, extractor_output_dim, extractor_use_bias, extractor_stride
    )
    emformer = _get_emformer_encoder(
        encoder_input_dim,
        encoder_output_dim,
        encoder_num_heads,
        encoder_ffn_dim,
        encoder_num_layers,
        encoder_segment_length,
        encoder_left_context_length,
        encoder_right_context_length,
        encoder_dropout,
        encoder_activation,
        encoder_max_memory_size,
        encoder_weight_init_scale_strategy,
        encoder_tanh_on_mem,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_output_dim, out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, emformer, aux)


def emformer_hubert_base(
    extractor_input_dim: int = 80,
    extractor_output_dim: int = 128,
    encoder_dropout: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Build Emformer HuBERT Model with 20 Emformer layers.

    Args:
        extractor_input_dim (int, optional): The input dimension for feature extractor. (Default: 80)
        extractor_output_dim (int, optional): The output dimension after feature extractor. (Default: 128)
        encoder_dropout (float, optional): Dropout probability in Emformer. (Default: 0.1)
        aux_num_out (int or None, optional): Output dimension of aux layer for fine-tuning. (Default: ``None``)

    Returns:
        Wav2Vec2Model:
            The resulting :py:class:`torchaudio.models.Wav2Vec2Model` model
            with a :py:class:`torchaudio.models.Emformer` encoder.
    """
    return emformer_hubert_model(
        extractor_input_dim=extractor_input_dim,
        extractor_output_dim=extractor_output_dim,
        extractor_use_bias=False,
        extractor_stride=4,
        encoder_input_dim=512,
        encoder_output_dim=1024,
        encoder_num_heads=8,
        encoder_ffn_dim=2048,
        encoder_num_layers=20,
        encoder_segment_length=4,
        encoder_left_context_length=30,
        encoder_right_context_length=1,
        encoder_dropout=encoder_dropout,
        encoder_activation="gelu",
        encoder_max_memory_size=0,
        encoder_weight_init_scale_strategy="depthwise",
        encoder_tanh_on_mem=True,
        aux_num_out=aux_num_out,
    )
