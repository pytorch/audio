import torch
from pytorch_lightning import LightningDataModule
from torchaudio.datasets.librispeech import LIBRISPEECH

from ._utils import BucketizeBatchSampler, CollateFnWav2Vec2, DistributedBatchSampler


class Wav2Vec2DataModule(LightningDataModule):
    librispeech_cls = LIBRISPEECH

    def __init__(
        self,
        *,
        dataset_path,
        seconds_per_batch,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.seconds_per_batch = seconds_per_batch
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                self.librispeech_cls(self.dataset_path, url="train-clean-360"),
                self.librispeech_cls(self.dataset_path, url="train-clean-100"),
                self.librispeech_cls(self.dataset_path, url="train-other-500"),
            ]
        )
        len_list = [d[0].size(1) for d in dataset]

        sampler = BucketizeBatchSampler(
            len_list,
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
            collate_fn=CollateFnWav2Vec2(pad=False, rand_crop=True),
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                self.librispeech_cls(self.librispeech_path, url="dev-clean"),
                self.librispeech_cls(self.librispeech_path, url="dev-other"),
            ]
        )
        len_list = [d[0].size(1) for d in dataset]
        sampler = BucketizeBatchSampler(
            len_list,
            num_buckets=1000,
            max_token_count=self.seconds_per_batch * 16000,
            min_len=32000,
            max_len=250000,
            shuffle=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=CollateFnWav2Vec2(pad=False, rand_crop=True),
            num_workers=self.num_workers,
        )
        return dataloader
