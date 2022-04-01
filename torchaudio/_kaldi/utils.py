import torch
from torch import Tensor


def _get_num_hops(num_frames, window_size, window_shift, snip_edges):
    if snip_edges:
        if num_frames < window_size:
            return 0
        return 1 + (num_frames - window_size) // window_shift

    # The added window_shift // 2 is to so that the result of the
    # final division become the closest integer of `num_frames / window_shift`
    return (num_frames + (window_shift // 2)) // window_shift


def fold(
        waveform: Tensor,
        window_size: int,
        window_shift: int,
        snip_edges: bool,
) -> Tensor:
    """Extract windows along timeaxis and batch them together

    .. code:

       ------------------------------------------->
         1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ...


       -------------------------------------------->
         1     4     7     10       13  ...
         2     5     8     11       14
         3     6     9     12       15

    """
    assert waveform.ndim == 2
    num_frames = waveform.shape[1]

    # The index of the first frame and the number of hops
    #
    # If snip_edges == True, then the fold starts at index 0 and all folds contain
    # valid frames. The remaining frames at the end are dropped.
    #
    # If snip_edges == False, then the fold starts so that the middle point of the
    # first fold is centered at offset of shift // 2. The number of hops are an
    # integer closest to num_frames // window_shift.
    #
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-window.cc#L30
    i_start = 0 if snip_edges else window_shift // 2 - window_size // 2
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-window.cc#L42
    num_hops = _get_num_hops(num_frames, window_size, window_shift, snip_edges)

    i_end = i_start + num_hops * window_shift + window_size

    if i_start < 0 or num_frames < i_end:
        # extraction starts/ends at outside of existing frames.
        left_pad = - i_start if i_start < 0 else 0
        right_pad = i_end - num_frames if num_frames < i_end else 0
        pad = (0, 0, left_pad, right_pad)
        waveform = torch.nn.functional.pad(waveform, pad, mode='reflect')

        if i_start < 0:
            i_end -= i_start
            i_start -= i_start

    folds = []
    waveform = waveform.unsqueeze(1)
    for i in range(num_hops):
        s = i_start + i * window_shift
        e = s + window_size
        folds.append(waveform[:, :, s:e])
    return torch.cat(folds, dim=1)
