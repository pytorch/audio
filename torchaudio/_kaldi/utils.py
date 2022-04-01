import torch
from torch import Tensor


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
    if window_size <= 0:
        raise ValueError(f"'window_size' must be positive. Found: {window_size}")
    if  window_shift <= 0:
        raise ValueError(f"'window_shift' must be positive. Found: {window_shift}")
    if waveform.ndim != 2:
        raise ValueError(f"Expected 2D waveform. Shape found: {waveform.shape}")
    if waveform.shape[1] <= 0:
        raise ValueError(f"The input waveform cannot be empty. Shape found: {waveform.shape}")

    batch_size, num_frames = waveform.shape
    device, dtype = waveform.device, waveform.dtype

    # Figure out the index of the first window frame and the number of windows (== number of output frames)
    #
    # This is the simplification of the following two functions.
    #
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-window.cc#L30
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-window.cc#L42
    #
    # If snip_edges == True, then the first window starts at index 0 and all windows contain
    # valid frames. The remaining frames at the end are dropped.
    #
    # If snip_edges == False, then the first window starts so that the middle point of the
    # first window is at `window_shift // 2`.
    # The number of windows are an integer closest to num_frames // window_shift.
    #
    if snip_edges:
        i_start = 0
        num_wins = 0 if num_frames < window_size else 1 + (num_frames - window_size) // window_shift
    else:
        i_start = window_shift // 2 - window_size // 2
        num_wins = (num_frames + (window_shift // 2)) // window_shift

    i_end = i_start + window_size + (num_wins - 1) * window_shift
    # ~= num_frames + window_shift + window_size // 2 (when snip_edges == false)

    if i_start < - (num_frames + 1) or i_end > num_frames * 2:
        raise ValueError(
            "The requested window region is outside of the valid range. "
            "The input waveform is too short or the configuration is invalid. "
            "Perhaps, set snip_edges=True, or "
            "decrease the value of window_size and/or window_shift."
        )
    # assert i_end > 0

    if num_wins == 0:
        # TODO: issue warning?
        return torch.empty([batch_size, 0, window_size], device=device, dtype=dtype)

    # if i_start < 0 or num_frames < i_end:
    #     # extraction starts/ends at outside of existing frames.
    #     left_pad = - i_start if i_start < 0 else 0
    #     right_pad = i_end - num_frames if num_frames < i_end else 0
    #     pad = (0, 0, left_pad, right_pad)
    #     waveform = torch.nn.functional.pad(waveform, pad, mode='reflect')
    #
    #     if i_start < 0:
    #         i_end -= i_start
    #         i_start -= i_start

    folds = []
    waveform = waveform.unsqueeze(1)

    # Note:
    # torch.nn.functional.pad(mode="reflect") performs fixed end reflection.
    # i.e. `1 2 3 2 1`,
    # while Kaldi's implementation performs free end reflection.
    # i.e. `1 2 3 3 2 1`.
    #
    # Kaldi's implementation allows multiple reflection if window length is
    # much larger than number of available frames, even though this is
    # mal-formed configuration.
    # In this implementation, such configuration is not allowed.
    for i in range(num_wins):
        s = i_start + i * window_shift
        e = s + window_size
        # e >= i_start + windows_size >= window_shift // 2 + window_size // 2 > 0
        if s < 0:
            #                       s           e
            # tensor index      -4 -3 -2 -1 0 1 2 3 4
            #                      ^^^^^^^^^^^^ x
            # corresponding      3  2  1  0 0 1 2 3 4
            # mirrored index       ^^^^^^^^ ^^^
            #
            # resulting slice   torch.cat(waveform[:, :, :3].flip(-1), waveform[:, :, :2])
            fold = torch.cat(waveform[:, :, :-s].flip(-1), waveform[:, :, :e], dim=-1)
        elif s <= num_frames:
            if e <= num_frames:
                #                     s     e
                # tensor index      0 1 2 3 4
                #                     ^^^^^ x
                fold = waveform[:, :, s:e]
            else:
                #            num_frames == 7
                #                     s          |       e
                # tensor index      0 1 2 3 4 5 6 7 8 9 10
                #                     ^^^^^^^^^^^^^^^^^  x
                # corresponding     0 1 2 3 4 5 6 6 5 4  3
                # mirrored index      ^^^^^^^^^^^ ^^^^^  x
                #
                # resulting slice   torch.cat(waveform[:, :, 1:], waveform[:, :, 4:].flip(-1))
                fold = torch.cat(waveform[:, :, s:], waveform[:, :, e - num_frames:].flip(-1), dim=-1)
        else:
            #            num_frames == 4
            #                          |  s   e
            # tensor index      0 1 2 3 4 5 6 7
            #                             ^^^ x
            # corresponding     0 1 2 3 3 2 1 0
            # mirrored index              ^^^ x
            #
            # resulting slice   torch.cat(waveform[:, :, 1:3].flip(-1))
            fold = torch.cat(waveform[:, :, s - num_frames:e - num_frames].flip(-1), dim=-1)
        folds.append(fold)
    return torch.cat(folds, dim=1)
