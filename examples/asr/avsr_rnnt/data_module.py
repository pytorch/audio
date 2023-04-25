import os
import random

import torch
import torchaudio
from pytorch_lightning import LightningDataModule

from lrs3 import LRS3


def _batch_by_token_count(idx_target_lengths, max_frames, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_frames or (batch_size and len(current_batch) == batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        max_frames,
        num_buckets,
        shuffle=False,
        batch_size=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

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
            max_frames,
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


class LRS3DataModule(LightningDataModule):

    def __init__(
        self,
        *,
        args,
        train_transform,
        val_transform,
        test_transform,
        max_frames,
        batch_size=None,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.args = args
        self.train_dataset_lengths = None
        self.val_dataset_lengths = None
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        datasets = [
            LRS3(self.args, subset="train")
        ]

        if not self.train_dataset_lengths:
            self.train_dataset_lengths = [dataset._lengthlist for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_frames,
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
            LRS3(self.args, subset="val")
        ]

        if not self.val_dataset_lengths:
            self.val_dataset_lengths = [dataset._lengthlist for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_frames,
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
        dataset = LRS3(self.args, subset="test")
        dataset = TransformDataset(dataset, self.test_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
