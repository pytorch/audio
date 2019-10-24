import math

import torch
import torchaudio


def compute_correllogram(waveform, sample_rate, frame_time=10 ** -2):
    """
    Compute correlogram.
    """

    # https://en.wikipedia.org/wiki/Voice_frequency
    # Usable voice frequencies for telephony: 30-3400 Hz
    # Fundamental Frequency: 85-180 Hz or 165-255 Hz

    # Number of lags to check
    lags = math.ceil(sample_rate / 85)  # Around 500 samples

    frame_size = int(math.ceil(sample_rate * frame_time))
    EPSILON = 10 ** (-9)

    w = waveform.view(-1)
    waveform_length = w.size()[-1]
    num_of_frames = math.ceil(waveform_length / frame_size)

    p = lags + num_of_frames * frame_size - waveform_length
    w = torch.nn.functional.pad(w, (0, p))

    # Compute lags
    output_lag = []
    for lag in range(1, lags + 1):
        s1 = w[:-lag].unfold(0, frame_size, frame_size)[:num_of_frames, :]
        s2 = w[lag:].unfold(0, frame_size, frame_size)[:num_of_frames, :]

        output_frames = (
            (s1 * s2).sum(-1)
            / (EPSILON + s1.norm(dim=-1))
            / (EPSILON + s2.norm(dim=-1))
        )

        output_lag.append(output_frames.view(-1, 1))

    correlogram = torch.cat(output_lag, 1)

    return correlogram, frame_size


def _combine_max(a, b, thresh=0.99):
    mask = (a[0] > thresh * b[0]).to(int)
    values = mask * a[0] + (1 - mask) * b[0]
    indices = mask * a[1] + (1 - mask) * b[1]
    return values, indices


def find_max_per_frame(correlogram, sample_rate, smoothing_window=30):
    EPSILON = 10 ** (-9)

    # https://en.wikipedia.org/wiki/Voice_frequency
    # Usable voice frequencies for telephony: 30-3400 Hz
    # Fundamental Frequency: 85-180 Hz or 165-255 Hz

    # Voice frequency is no shorter
    lag_min = math.ceil(sample_rate / 3400)  # Around 10 samples

    # Find near enough max that is smallest

    best = torch.max(correlogram[:, lag_min:], -1)
    half = torch.max(correlogram[:, lag_min: correlogram.shape[-1] // 2], -1)
    # quarter = torch.max(correlogram[:, lag_min: correlogram.shape[-1] // 4], -1)

    best = _combine_max(half, best)
    # best = _combine_max(quarter, best)

    indices = best[1]

    # Add back minimal lag
    indices += lag_min
    # Add 1 empirical calibration offset
    indices += 1

    # Median smoothing

    # Online
    # indices = torch.nn.functional.pad(indices, (smoothing_window-1,0), mode="constant", value=indices[0])
    # Centered
    indices = torch.nn.functional.pad(
        indices, ((smoothing_window - 1) // 2, 0), mode="constant", value=indices[0]
    )
    roll = indices.unfold(0, smoothing_window, 1)
    values, _ = torch.median(roll, -1)
    freq = sample_rate / (EPSILON + values.to(torch.float))

    return freq


def reconstruct_waveform(frames, waveform, frame_size):
    z = torch.zeros(waveform.shape, dtype=waveform.dtype, device=waveform.device)

    # Convert to waveform
    n_frames = len(frames)
    for i in range(n_frames):
        z[0, i * frame_size: (i + 1) * frame_size] = frames[i]

    # Extend with what was last detected
    z[0, n_frames * frame_size:] = frames[-1]
    return z


if __name__ == "__main__":

    # Files from https://www.mediacollege.com/audio/tone/download/
    tests = [
        ("100Hz_44100Hz_16bit_05sec.wav", 100),
        ("440Hz_44100Hz_16bit_05sec.wav", 440),
    ]

    for filename, freq_ref in tests:
        waveform, sample_rate = torchaudio.load(filename)
        waveform = waveform.mean(0).view(1, -1)

        correlogram, frame_size = compute_correllogram(waveform, sample_rate)
        freq = find_max_per_frame(correlogram, sample_rate)
        # freq = reconstruct_waveform(freq, waveform, frame_size)

        threshold = 1
        print(((freq - freq_ref).abs() > threshold).sum())
