from pathlib import Path
from typing import (
    Union,
)
import logging
from torch import Tensor
import torch
import torchaudio

logger = logging.getLogger("feature_utils")


def get_shard_range(tot, num_rank, rank):
    assert rank < num_rank and rank >= 0, f"invaid rank/num_rank {rank}/{num_rank}"
    start = round(tot / num_rank * rank)
    end = round(tot / num_rank * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {num_rank}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def extract_feature(
    path: str,
    device: torch.device,
    feature_type: str,
    sample_rate: int,
) -> Tensor:
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    waveform = waveform[0].to(device)
    if feature_type == "mfcc":
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate
        ).to(device)
        mfccs = feature_extractor(waveform)  # (freq, time)
        # mfccs = torchaudio.compliance.kaldi.mfcc(
        #     waveform=waveform,
        #     sample_frequency=sample_rate,
        #     use_energy=False,
        # )  # (time, freq)
        # mfccs = mfccs.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1)  # (time, freq)
        return concat
    else:
        raise Exception("Not implemented yet")


def dump_features(
    tsv_file: Union[str, Path],
    out_dir: Union[str, Path],
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    feature_type: str = "mfcc",
    sample_rate: int = 16_000,
) -> None:
    features = []
    lens = []
    out_dir = Path(out_dir)

    feature_path = out_dir / f"{split}_{rank}_{num_rank}.pt"
    len_path = out_dir / f"len_{split}_{rank}_{num_rank}.pt"
    with open(tsv_file, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), num_rank, rank)
        lines = lines[start:end]
        for line in lines:
            path, nsample = line.split("\t")
            path = f"{root}/{path}"
            nsample = int(nsample)
            feature = extract_feature(path, device, feature_type, sample_rate)
            features.append(feature.cpu())
            lens.append(feature.shape[0])
    features = torch.cat(features)
    lens = torch.Tensor(lens)
    torch.save(features, feature_path)
    torch.save(lens, len_path)
    logger.info(f"Finished rank {rank} of {num_rank} successfully")
