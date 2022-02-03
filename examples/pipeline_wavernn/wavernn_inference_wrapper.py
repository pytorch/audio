# *****************************************************************************
# Copyright (c) 2019 fatchord (https://github.com/fatchord)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *****************************************************************************


import torch
import torchaudio
from processing import normalized_waveform_to_bits
from torch import Tensor
from torchaudio.models.wavernn import WaveRNN


def _fold_with_overlap(x: Tensor, timesteps: int, overlap: int) -> Tensor:
    r"""Fold the tensor with overlap for quick batched inference.
    Overlap will be used for crossfading in xfade_and_unfold().

    x = [[h1, h2, ... hn]]
    Where each h is a vector of conditioning channels
    Eg: timesteps=2, overlap=1 with x.size(1)=10
    folded = [[h1, h2, h3, h4],
              [h4, h5, h6, h7],
              [h7, h8, h9, h10]]

    Args:
        x (tensor): Upsampled conditioning channels of size (1, timesteps, channel).
        timesteps (int): Timesteps for each index of batch.
        overlap (int): Timesteps for both xfade and rnn warmup.

    Return:
        folded (tensor): folded tensor of size (n_folds, timesteps + 2 * overlap, channel).
    """

    _, channels, total_len = x.size()

    # Calculate variables needed
    n_folds = (total_len - overlap) // (timesteps + overlap)
    extended_len = n_folds * (overlap + timesteps) + overlap
    remaining = total_len - extended_len

    # Pad if some time steps poking out
    if remaining != 0:
        n_folds += 1
        padding = timesteps + 2 * overlap - remaining
        x = torch.nn.functional.pad(x, (0, padding))

    folded = torch.zeros((n_folds, channels, timesteps + 2 * overlap), device=x.device)

    # Get the values for the folded tensor
    for i in range(n_folds):
        start = i * (timesteps + overlap)
        end = start + timesteps + 2 * overlap
        folded[i] = x[0, :, start:end]

    return folded


def _xfade_and_unfold(y: Tensor, overlap: int) -> Tensor:
    r"""Applies a crossfade and unfolds into a 1d array.

    y = [[seq1],
         [seq2],
         [seq3]]
    Apply a gain envelope at both ends of the sequences
    y = [[seq1_in, seq1_timesteps, seq1_out],
         [seq2_in, seq2_timesteps, seq2_out],
         [seq3_in, seq3_timesteps, seq3_out]]
    Stagger and add up the groups of samples:
        [seq1_in, seq1_timesteps, (seq1_out + seq2_in), seq2_timesteps, ...]

    Args:
        y (Tensor): Batched sequences of audio samples of size
            (num_folds, channels, timesteps + 2 * overlap).
        overlap (int): Timesteps for both xfade and rnn warmup.

    Returns:
        unfolded waveform (Tensor) : waveform in a 1d tensor of size (channels, total_len).
    """

    num_folds, channels, length = y.shape
    timesteps = length - 2 * overlap
    total_len = num_folds * (timesteps + overlap) + overlap

    # Need some silence for the rnn warmup
    silence_len = overlap // 2
    fade_len = overlap - silence_len
    silence = torch.zeros((silence_len), dtype=y.dtype, device=y.device)
    linear = torch.ones((silence_len), dtype=y.dtype, device=y.device)

    # Equal power crossfade
    t = torch.linspace(-1, 1, fade_len, dtype=y.dtype, device=y.device)
    fade_in = torch.sqrt(0.5 * (1 + t))
    fade_out = torch.sqrt(0.5 * (1 - t))

    # Concat the silence to the fades
    fade_in = torch.cat([silence, fade_in])
    fade_out = torch.cat([linear, fade_out])

    # Apply the gain to the overlap samples
    y[:, :, :overlap] *= fade_in
    y[:, :, -overlap:] *= fade_out

    unfolded = torch.zeros((channels, total_len), dtype=y.dtype, device=y.device)

    # Loop to add up all the samples
    for i in range(num_folds):
        start = i * (timesteps + overlap)
        end = start + timesteps + 2 * overlap
        unfolded[:, start:end] += y[i]

    return unfolded


class WaveRNNInferenceWrapper(torch.nn.Module):
    def __init__(self, wavernn: WaveRNN):
        super().__init__()
        self.wavernn_model = wavernn

    def forward(
        self, specgram: Tensor, mulaw: bool = True, batched: bool = True, timesteps: int = 100, overlap: int = 5
    ) -> Tensor:
        r"""Inference function for WaveRNN.

        Based on the implementation from
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py.


        Currently only supports multinomial sampling.

        Args:
            specgram (Tensor): spectrogram of size (n_mels, n_time)
            mulaw (bool, optional): Whether to perform mulaw decoding (Default: ``True``).
            batched (bool, optional): Whether to perform batch prediction. Using batch prediction
                will significantly increase the inference speed (Default: ``True``).
            timesteps (int, optional): The time steps for each batch. Only used when `batched`
                is set to True (Default: ``100``).
            overlap (int, optional): The overlapping time steps between batches. Only used when
                `batched` is set to True (Default: ``5``).

        Returns:
            waveform (Tensor): Reconstructed waveform of size (1, n_time, ).
                1 represents single channel.
        """
        specgram = specgram.unsqueeze(0)
        if batched:
            specgram = _fold_with_overlap(specgram, timesteps, overlap)

        output = self.wavernn_model.infer(specgram).cpu()

        if mulaw:
            output = normalized_waveform_to_bits(output, self.wavernn_model.n_bits)
            output = torchaudio.functional.mu_law_decoding(output, self.wavernn_model.n_classes)

        if batched:
            output = _xfade_and_unfold(output, overlap)
        else:
            output = output[0]

        return output
