import os
import random

import torch
import torchaudio
from pytorch_lightning import LightningDataModule


def _batch_by_token_count(idx_target_lengths, max_tokens, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_tokens or (batch_size and len(current_batch) == batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


def get_sample_lengths(librispeech_dataset):
    fileid_to_target_length = {}

    def _target_length(fileid):
        if fileid not in fileid_to_target_length:
            speaker_id, chapter_id, _ = fileid.split("-")

            file_text = speaker_id + "-" + chapter_id + librispeech_dataset._ext_txt
            file_text = os.path.join(librispeech_dataset._path, speaker_id, chapter_id, file_text)

            with open(file_text) as ft:
                for line in ft:
                    fileid_text, transcript = line.strip().split(" ", 1)
                    fileid_to_target_length[fileid_text] = len(transcript)

        return fileid_to_target_length[fileid]

    return [_target_length(fileid) for fileid in librispeech_dataset._walker]


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        max_tokens,
        num_buckets,
        shuffle=False,
        batch_size=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_tokens >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_tokens,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class LibriSpeechDataModule(LightningDataModule):
    librispeech_cls = torchaudio.datasets.LIBRISPEECH_BIASING

    def __init__(
        self,
        *,
        librispeech_path,
        train_transform,
        val_transform,
        test_transform,
        max_tokens=3200,
        batch_size=16,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=10,
        subset=None,
        fullbiasinglist=[]
    ):
        super().__init__()
        self.librispeech_path = librispeech_path
        self.train_dataset_lengths = None
        self.val_dataset_lengths = None
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.subset = subset
        self.fullbiasinglist = fullbiasinglist

    def train_dataloader(self):
        if self.subset == "clean100":
            datasets = [self.librispeech_cls(self.librispeech_path, url="train-clean-100", blist=self.fullbiasinglist)]
        else:
            datasets = [
                self.librispeech_cls(self.librispeech_path, url="train-clean-360"),
                self.librispeech_cls(self.librispeech_path, url="train-clean-100"),
                self.librispeech_cls(self.librispeech_path, url="train-other-500"),
            ]

        if not self.train_dataset_lengths:
            self.train_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    self.train_num_buckets,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.train_dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.train_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    def val_dataloader(self):
        datasets = [
            self.librispeech_cls(self.librispeech_path, url="dev-clean", blist=self.fullbiasinglist),
            self.librispeech_cls(self.librispeech_path, url="dev-other", blist=self.fullbiasinglist),
        ]

        if not self.val_dataset_lengths:
            self.val_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    1,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.val_dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.val_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = self.librispeech_cls(self.librispeech_path, url="test-clean", blist=self.fullbiasinglist)
        dataset = TransformDataset(dataset, self.test_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
