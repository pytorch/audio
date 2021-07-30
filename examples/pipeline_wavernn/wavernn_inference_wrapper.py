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


from typing import List

from torchaudio.models.wavernn import WaveRNN
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from processing import (
    normalized_waveform_to_bits,
    bits_to_normalized_waveform,
)

class WaveRNNInferenceWrapper(torch.nn.Module):

    def __init__(self, wavernn: WaveRNN):
        super().__init__()
        self.wavernn_model = wavernn

    def _fold_with_overlap(self, x: Tensor, timesteps: int, overlap: int) -> Tensor:
        r'''Fold the tensor with overlap for quick batched inference.
        Overlap will be used for crossfading in xfade_and_unfold().

        x = [[h1, h2, ... hn]]
        Where each h is a vector of conditioning channels
        Eg: timesteps=2, overlap=1 with x.size(1)=10
        folded = [[h1, h2, h3, h4],
                  [h4, h5, h6, h7],
                  [h7, h8, h9, h10]]

        Args:
            x (tensor): Upsampled conditioning channels with shape (1, timesteps, channel).
            timesteps (int): Timesteps for each index of batch.
            overlap (int): Timesteps for both xfade and rnn warmup.

        Return:
            folded (tensor): folded tensor with shape (n_folds, timesteps + 2 * overlap, channel).
        '''

        _, channels, total_len = x.size()

        # Calculate variables needed
        n_folds = (total_len - overlap) // (timesteps + overlap)
        extended_len = n_folds * (overlap + timesteps) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            n_folds += 1
            padding = timesteps + 2 * overlap - remaining
            x = self._pad_tensor(x, padding, side='after')

        folded = torch.zeros((n_folds, channels, timesteps + 2 * overlap), device=x.device)

        # Get the values for the folded tensor
        for i in range(n_folds):
            start = i * (timesteps + overlap)
            end = start + timesteps + 2 * overlap
            folded[i] = x[0, :, start:end]

        return folded

    def _xfade_and_unfold(self, y: Tensor, overlap: int) -> Tensor:
        r'''Applies a crossfade and unfolds into a 1d array.

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
            y (Tensor): Batched sequences of audio samples with shape
                (num_folds, timesteps + 2 * overlap).
            overlap (int): Timesteps for both xfade and rnn warmup.

        Returns:
            unfolded waveform (Tensor) : waveform in a 1d tensor with shape (total_len).
        '''

        num_folds, length = y.shape
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
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = torch.zeros((total_len), dtype=y.dtype, device=y.device)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (timesteps + overlap)
            end = start + timesteps + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def _pad_tensor(self, x: Tensor, pad: int, side: str = 'both') -> Tensor:
        r"""Pad the given tensor.

        Args:
            x (Tensor): The tensor to pad with shape (n_batch, n_mels, time).
            pad (int): The amount of padding applied to the input.

        Return:
            padded (Tensor): The padded tensor with shape (n_batch, n_mels, time).
        """
        b, c, t = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros((b, c, total), device=x.device)
        if side == 'before' or side == 'both':
            padded[:, :, pad:pad + t] = x
        elif side == 'after':
            padded[:, :, :t] = x
        else:
            raise ValueError(f"Unexpected side: '{side}'. "
                            f"Valid choices are 'both', 'before' and 'after'.")
        return padded

    def forward(self,
                specgram: Tensor,
                loss_name: str = "crossentropy",
                mulaw: bool = True,
                batched: bool = True,
                timesteps: int = 11000,
                overlap: int = 550) -> Tensor:
        r"""Inference function for WaveRNN.

        Based on the implementation from
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py.

        Args:
            specgram (Tensor): spectrogram with shape (n_mels, n_time)
            loss_name (str): The loss function used to train the WaveRNN model.
                Available `loss_name` includes `'mol'` and `'crossentropy'`.
            mulaw (bool): Whether to perform mulaw decoding (Default: ``True``).
            batched (bool): Whether to perform batch prediction. Using batch prediction
                will significantly increase the inference speed (Default: ``True``).
            timesteps (int): The time steps for each batch. Only used when `batched`
                is set to True (Default: ``11000``).
            overlap (int): The overlapping time steps between batches. Only used when `batched`
                is set to True (Default: ``550``).

        Returns:
            waveform (Tensor): Reconstructed waveform with shape (n_time, ).
        """
        pad = (self.wavernn_model.kernel_size - 1) // 2

        specgram = specgram.unsqueeze(0)
        specgram = self._pad_tensor(specgram, pad=pad, side='both')
        specgram, aux = self.wavernn_model.upsample(specgram)

        if batched:
            specgram = self._fold_with_overlap(specgram, timesteps, overlap)
            aux = self._fold_with_overlap(aux, timesteps, overlap)

        device = specgram.device
        dtype = specgram.dtype
        # make it compatible with torchscript
        n_bits = int(torch.log2(torch.ones(1) * self.wavernn_model.n_classes))
        output: List[Tensor] = []
        b_size, _, seq_len = specgram.size()

        h1 = torch.zeros((1, b_size, self.wavernn_model.n_rnn), device=device, dtype=dtype)
        h2 = torch.zeros((1, b_size, self.wavernn_model.n_rnn), device=device, dtype=dtype)
        x = torch.zeros((b_size, 1), device=device, dtype=dtype)

        d = self.wavernn_model.n_aux
        aux_split = [aux[:, d * i:d * (i + 1), :] for i in range(4)]

        for i in range(seq_len):

            m_t = specgram[:, :, i]

            a1_t, a2_t, a3_t, a4_t = [a[:, :, i] for a in aux_split]

            x = torch.cat([x, m_t, a1_t], dim=1)
            x = self.wavernn_model.fc(x)
            _, h1 = self.wavernn_model.rnn1(x.unsqueeze(1), h1)

            x = x + h1[0]
            inp = torch.cat([x, a2_t], dim=1)
            _, h2 = self.wavernn_model.rnn2(inp.unsqueeze(1), h2)

            x = x + h2[0]
            x = torch.cat([x, a3_t], dim=1)
            x = F.relu(self.wavernn_model.fc1(x))

            x = torch.cat([x, a4_t], dim=1)
            x = F.relu(self.wavernn_model.fc2(x))

            logits = self.wavernn_model.fc3(x)

            if loss_name == "crossentropy":
                posterior = F.softmax(logits, dim=1)

                x = torch.multinomial(posterior, 1).float()
                x = bits_to_normalized_waveform(x, n_bits)
                output.append(x.squeeze(-1))
            else:
                raise ValueError(f"Unexpected loss_name: '{loss_name}'. "
                                f"Valid choices are 'crossentropy'.")

        output = torch.stack(output).transpose(0, 1).cpu()

        if mulaw:
            output = normalized_waveform_to_bits(output, n_bits)
            output = torchaudio.functional.mu_law_decoding(output, self.wavernn_model.n_classes)

        if batched:
            output = self._xfade_and_unfold(output, overlap)
        else:
            output = output[0]

        return output
