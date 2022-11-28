import torch
from pytorch_lightning import LightningDataModule

from ._utils import (
    BucketizeBatchSampler,
    CollateFnHubert,
    DistributedBatchSampler,
    HuBERTDataSet,
)


class HuBERTDataModule(LightningDataModule):
    hubert_cls = HuBERTDataSet

    def __init__(
        self,
        *,
        dataset_path,
        dataset,
        feature_type,
        seconds_per_batch,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.feature_type = feature_type
        self.seconds_per_batch = seconds_per_batch
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = self.hubert_cls(self.dataset_path, self.dataset, "train")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=10000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=True,
        )
        sampler = DistributedBatchSampler(sampler, shuffle=self.train_shuffle)
        sampler.set_epoch(self.trainer.current_epoch)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(
                feature_type=self.feature_type, pad=False, rand_crop=True
            ),
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataset = self.hubert_cls(self.dataset_path, self.dataset, "valid")
        sampler = BucketizeBatchSampler(
            dataset.len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnHubert(
                feature_type=self.feature_type, pad=False, rand_crop=True
            ),
            num_workers=self.num_workers,
        )
        return dataloader
