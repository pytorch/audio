import math
from typing import Dict, List, Optional, Tuple

import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber


TrieNode = Tuple[Dict[int, "TrieNode"], int, Optional[Tuple[int, int]]]


class _ConformerEncoder(torch.nn.Module, _Transcriber):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        time_reduction_stride: int,
        conformer_input_dim: int,
        conformer_ffn_dim: int,
        conformer_num_layers: int,
        conformer_num_heads: int,
        conformer_depthwise_conv_kernel_size: int,
        conformer_dropout: float,
    ) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = torch.nn.Linear(input_dim * time_reduction_stride, conformer_input_dim)
        self.conformer = Conformer(
            num_layers=conformer_num_layers,
            input_dim=conformer_input_dim,
            ffn_dim=conformer_ffn_dim,
            num_heads=conformer_num_heads,
            depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
            dropout=conformer_dropout,
            use_group_norm=True,
            convolution_first=True,
        )
        self.output_linear = torch.nn.Linear(conformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise RuntimeError("Conformer does not support streaming inference.")


class _JoinerBiasing(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        biasing (bool): perform biasing
        deepbiasing (bool): perform deep biasing
        attndim (int): dimension of the biasing vector hptr

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        biasing: bool = False,
        deepbiasing: bool = False,
        attndim: int = 1,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.biasing = biasing
        self.deepbiasing = deepbiasing
        if self.biasing and self.deepbiasing:
            self.biasinglinear = torch.nn.Linear(attndim, input_dim, bias=True)
            self.attndim = attndim
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
        hptr: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
            hptr (torch.Tensor): deep biasing vector with shape `(B, T, U, A)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                torch.Tensor
                    joint network second last layer output (i.e. before self.linear), with shape `(B, T, U, D)`.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        if self.biasing and self.deepbiasing and hptr is not None:
            hptr = self.biasinglinear(hptr)
            joint_encodings += hptr
        elif self.biasing and self.deepbiasing:
            # Hack here for unused parameters
            joint_encodings += self.biasinglinear(joint_encodings.new_zeros(1, self.attndim)).mean() * 0
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths, activation_out


class RNNTBiasing(RNNT):
    r"""torchaudio.models.RNNT()

    Recurrent neural network transducer (RNN-T) model.

    Note:
        To build the model, please use one of the factory functions.

    Args:
        transcriber (torch.nn.Module): transcription network.
        predictor (torch.nn.Module): prediction network.
        joiner (torch.nn.Module): joint network.
        attndim (int): TCPGen attention dimension
        biasing (bool): If true, use biasing, otherwise use standard RNN-T
        deepbiasing (bool): If true, use deep biasing by extracting the biasing vector
        embdim (int): dimension of symbol embeddings
        jointdim (int): dimension of the joint network joint dimension
        charlist (list): The list of word piece tokens in the same order as the output layer
        encoutdim (int): dimension of the encoder output vectors
        dropout_tcpgen (float): dropout rate for TCPGen
        tcpsche (int): The epoch at which TCPGen starts to train
        DBaverage (bool): If true, instead of TCPGen, use DBRNNT for biasing
    """

    def __init__(
        self,
        transcriber: _Transcriber,
        predictor: _Predictor,
        joiner: _Joiner,
        attndim: int,
        biasing: bool,
        deepbiasing: bool,
        embdim: int,
        jointdim: int,
        charlist: List[str],
        encoutdim: int,
        dropout_tcpgen: float,
        tcpsche: int,
        DBaverage: bool,
    ) -> None:
        super().__init__(transcriber, predictor, joiner)
        self.attndim = attndim
        self.deepbiasing = deepbiasing
        self.jointdim = jointdim
        self.embdim = embdim
        self.encoutdim = encoutdim
        self.char_list = charlist or []
        self.blank_idx = self.char_list.index("<blank>")
        self.nchars = len(self.char_list)
        self.DBaverage = DBaverage
        self.biasing = biasing
        if self.biasing:
            if self.deepbiasing and self.DBaverage:
                # Deep biasing without TCPGen
                self.biasingemb = torch.nn.Linear(self.nchars, self.attndim, bias=False)
            else:
                # TCPGen parameters
                self.ooKBemb = torch.nn.Embedding(1, self.embdim)
                self.Qproj_char = torch.nn.Linear(self.embdim, self.attndim)
                self.Qproj_acoustic = torch.nn.Linear(self.encoutdim, self.attndim)
                self.Kproj = torch.nn.Linear(self.embdim, self.attndim)
                self.pointer_gate = torch.nn.Linear(self.attndim + self.jointdim, 1)
        self.dropout_tcpgen = torch.nn.Dropout(dropout_tcpgen)
        self.tcpsche = tcpsche

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        tries: TrieNode,
        current_epoch: int,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            tries (TrieNode): wordpiece prefix trees representing the biasing list to be searched
            current_epoch (Int): the current epoch number to determine if TCPGen should be trained
                at this epoch
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
                torch.Tensor
                    TCPGen distribution, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    Generation probability (or copy probability), with shape
                    `(B, max output source length, max output target length, 1)`.
        """
        source_encodings, source_lengths = self.transcriber(
            input=sources,
            lengths=source_lengths,
        )
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=predictor_state,
        )
        # Forward TCPGen
        hptr = None
        tcpgen_dist, p_gen = None, None
        if self.biasing and current_epoch >= self.tcpsche and tries != []:
            ptrdist_mask, p_gen_mask = self.get_tcpgen_step_masks(targets, tries)
            hptr, tcpgen_dist = self.forward_tcpgen(targets, ptrdist_mask, source_encodings)
            hptr = self.dropout_tcpgen(hptr)
        elif self.biasing:
            # Hack here to bypass unused parameters
            if self.DBaverage and self.deepbiasing:
                dummy = self.biasingemb(source_encodings.new_zeros(1, len(self.char_list))).mean()
            else:
                dummy = source_encodings.new_zeros(1, self.embdim)
                dummy = self.Qproj_char(dummy).mean()
                dummy += self.Qproj_acoustic(source_encodings.new_zeros(1, source_encodings.size(-1))).mean()
                dummy += self.Kproj(source_encodings.new_zeros(1, self.embdim)).mean()
                dummy += self.pointer_gate(source_encodings.new_zeros(1, self.attndim + self.jointdim)).mean()
                dummy += self.ooKBemb.weight.mean()
            dummy = dummy * 0
            source_encodings += dummy

        output, source_lengths, target_lengths, jointer_activation = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
            hptr=hptr,
        )

        # Calculate Generation Probability
        if self.biasing and hptr is not None and tcpgen_dist is not None:
            p_gen = torch.sigmoid(self.pointer_gate(torch.cat((jointer_activation, hptr), dim=-1)))
            # avoid collapsing to ooKB token in the first few updates
            # if current_epoch == self.tcpsche:
            #     p_gen = p_gen * 0.1
            p_gen = p_gen.masked_fill(p_gen_mask.bool().unsqueeze(1).unsqueeze(-1), 0)

        return (output, source_lengths, target_lengths, predictor_state, tcpgen_dist, p_gen)

    def get_tcpgen_distribution(self, query, ptrdist_mask):
        # Make use of the predictor embedding matrix
        keyvalues = torch.cat([self.predictor.embedding.weight.data, self.ooKBemb.weight], dim=0)
        keyvalues = self.dropout_tcpgen(self.Kproj(keyvalues))
        # B * T * U * attndim, nbpe * attndim -> B * T * U * nbpe
        tcpgendist = torch.einsum("ntuj,ij->ntui", query, keyvalues)
        tcpgendist = tcpgendist / math.sqrt(query.size(-1))
        ptrdist_mask = ptrdist_mask.unsqueeze(1).repeat(1, tcpgendist.size(1), 1, 1)
        tcpgendist.masked_fill_(ptrdist_mask.bool(), -1e9)
        tcpgendist = torch.nn.functional.softmax(tcpgendist, dim=-1)
        # B * T * U * nbpe, nbpe * attndim -> B * T * U * attndim
        hptr = torch.einsum("ntui,ij->ntuj", tcpgendist[:, :, :, :-1], keyvalues[:-1, :])
        return hptr, tcpgendist

    def forward_tcpgen(self, targets, ptrdist_mask, source_encodings):
        tcpgen_dist = None
        if self.DBaverage and self.deepbiasing:
            hptr = self.biasingemb(1 - ptrdist_mask[:, :, :-1].float()).unsqueeze(1)
        else:
            query_char = self.predictor.embedding(targets)
            query_char = self.Qproj_char(query_char).unsqueeze(1)  # B * 1 * U * attndim
            query_acoustic = self.Qproj_acoustic(source_encodings).unsqueeze(2)  # B * T * 1 * attndim
            query = query_char + query_acoustic  # B * T * U * attndim
            hptr, tcpgen_dist = self.get_tcpgen_distribution(query, ptrdist_mask)
        return hptr, tcpgen_dist

    def get_tcpgen_step_masks(self, yseqs, resettrie):
        seqlen = len(yseqs[0])
        batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
        p_gen_masks = []
        for i, yseq in enumerate(yseqs):
            new_tree = resettrie
            p_gen_mask = []
            for j, vy in enumerate(yseq):
                vy = vy.item()
                new_tree = new_tree[0]
                if vy in [self.blank_idx]:
                    new_tree = resettrie
                    p_gen_mask.append(0)
                elif self.char_list[vy].endswith("▁"):
                    if vy in new_tree and new_tree[vy][0] != {}:
                        new_tree = new_tree[vy]
                    else:
                        new_tree = resettrie
                    p_gen_mask.append(0)
                elif vy not in new_tree:
                    new_tree = [{}]
                    p_gen_mask.append(1)
                else:
                    new_tree = new_tree[vy]
                    p_gen_mask.append(0)
                batch_masks[i, j, list(new_tree[0].keys())] = 0
                # In the original paper, ooKB node was not masked
                # In this implementation, if not masking ooKB, ooKB probability
                # would quickly collapse to 1.0 in the first few updates.
                # Haven't found out why this happened.
                # batch_masks[i, j, -1] = 0
            p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
        p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()
        return batch_masks, p_gen_masks

    def get_tcpgen_step_masks_prefix(self, yseqs, resettrie):
        # Implemented for prefix-based wordpieces, not tested yet
        seqlen = len(yseqs[0])
        batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
        p_gen_masks = []
        for i, yseq in enumerate(yseqs):
            p_gen_mask = []
            new_tree = resettrie
            for j, vy in enumerate(yseq):
                vy = vy.item()
                new_tree = new_tree[0]
                if vy in [self.blank_idx]:
                    new_tree = resettrie
                    batch_masks[i, j, list(new_tree[0].keys())] = 0
                elif self.char_list[vy].startswith("▁"):
                    new_tree = resettrie
                    if vy not in new_tree[0]:
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                    else:
                        new_tree = new_tree[0][vy]
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                        if new_tree[1] != -1:
                            batch_masks[i, j, list(resettrie[0].keys())] = 0
                else:
                    if vy not in new_tree:
                        new_tree = resettrie
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                    else:
                        new_tree = new_tree[vy]
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                        if new_tree[1] != -1:
                            batch_masks[i, j, list(resettrie[0].keys())] = 0
                p_gen_mask.append(0)
                # batch_masks[i, j, -1] = 0
            p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
        p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()

        return batch_masks, p_gen_masks

    def get_tcpgen_step(self, vy, trie, resettrie):
        new_tree = trie[0]
        if vy in [self.blank_idx]:
            new_tree = resettrie
        elif self.char_list[vy].endswith("▁"):
            if vy in new_tree and new_tree[vy][0] != {}:
                new_tree = new_tree[vy]
            else:
                new_tree = resettrie
        elif vy not in new_tree:
            new_tree = [{}]
        else:
            new_tree = new_tree[vy]
        return new_tree

    def join(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
        hptr: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Applies joint network to source and target encodings.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.
        A: TCPGen attention dimension

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
            hptr (torch.Tensor): deep biasing vector with shape `(B, T, U, A)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    joint network second last layer output, with shape `(B, T, U, D)`.
        """
        output, source_lengths, target_lengths, jointer_activation = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
            hptr=hptr,
        )
        return output, source_lengths, jointer_activation


def conformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
) -> RNNT:
    r"""Builds Conformer-based recurrent neural network transducer (RNN-T) model.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

        Returns:
            RNNT:
                Conformer RNN-T model.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols, activation=joiner_activation)
    return RNNT(encoder, predictor, joiner)


def conformer_rnnt_base() -> RNNT:
    r"""Builds basic version of Conformer RNN-T model.

    Returns:
        RNNT:
            Conformer RNN-T model.
    """
    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )


def conformer_rnnt_biasing(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
    attndim: int,
    biasing: bool,
    charlist: List[str],
    deepbiasing: bool,
    tcpsche: int,
    DBaverage: bool,
) -> RNNTBiasing:
    r"""Builds Conformer-based recurrent neural network transducer (RNN-T) model.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        attndim (int): TCPGen attention dimension
        biasing (bool): If true, use biasing, otherwise use standard RNN-T
        charlist (list): The list of word piece tokens in the same order as the output layer
        deepbiasing (bool): If true, use deep biasing by extracting the biasing vector
        tcpsche (int): The epoch at which TCPGen starts to train
        DBaverage (bool): If true, instead of TCPGen, use DBRNNT for biasing

        Returns:
            RNNT:
                Conformer RNN-T model with TCPGen-based biasing support.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _JoinerBiasing(
        encoding_dim,
        num_symbols,
        activation=joiner_activation,
        deepbiasing=deepbiasing,
        attndim=attndim,
        biasing=biasing,
    )
    return RNNTBiasing(
        encoder,
        predictor,
        joiner,
        attndim,
        biasing,
        deepbiasing,
        symbol_embedding_dim,
        encoding_dim,
        charlist,
        encoding_dim,
        conformer_dropout,
        tcpsche,
        DBaverage,
    )


def conformer_rnnt_biasing_base(charlist=None, biasing=True) -> RNNT:
    r"""Builds basic version of Conformer RNN-T model with TCPGen.

    Returns:
        RNNT:
            Conformer RNN-T model with TCPGen-based biasing support.
    """
    return conformer_rnnt_biasing(
        input_dim=80,
        encoding_dim=576,
        time_reduction_stride=4,
        conformer_input_dim=144,
        conformer_ffn_dim=576,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=601,
        symbol_embedding_dim=256,
        num_lstm_layers=1,
        lstm_hidden_dim=320,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
        attndim=256,
        biasing=biasing,
        charlist=charlist,
        deepbiasing=True,
        tcpsche=30,
        DBaverage=False,
    )
