from pathlib import Path
from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from torch import Tensor
import random
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


class BucketizeSampler(Sampler):
    """Buketize sampler for data with different lengths to reduce number of paddings.

    Args:
        data_source (Dataset): The dataset to sample
        num_buckets (int): The number of buckets to split the data samples.
        max_token_count (int or None, optional): The max number of tokens in one mini-batch.
            (Default: ``None``)
        batch_size (int or None, optional): The number of samples in one mini-batch.
             (Default: ``None``)

    Note: If ``max_token_count`` is not ``None``, the ``batch_size`` couldn't be set since
        the lengths of samples are unknown, the batch size may be different for different
        mini-batches.
    """
    def __init__(
        self,
        data_source: Dataset,
        num_buckets: int,
        max_token_count: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        if max_token_count is not None and batch_size is not None:
            raise AssertionError(
                "The ``max_token_count`` and ``batch_size`` can't be both set."
            )
        self.data_source = data_source
        self.max_token_count = max_token_count
        self.buckets = self._get_buckets(self.data_source, num_buckets)

    def _get_buckets(
        self,
        data_source: Dataset,
        num_buckets: int
    ) -> List[int]:
        buckets = {}
        len_list = data_source.len_list
        min_len, max_len = min(len_list), max(len_list)

        boundries = [min_len - 1]
        interval = (max_len - min_len) // num_buckets
        for i in range(1, num_buckets):
            boundries.append(min_len + i * interval)
        boundries.append(max_len + 1)
        bucket_ids = torch.bucketize(torch.Tensor(len_list), torch.Tensor(boundries))
        for i, length in enumerate(len_list):
            bucket_id = bucket_ids[i]
            if bucket_id in buckets:
                buckets[bucket_id].append(i)
            else:
                buckets[bucket_id] = [i]
        for k in buckets:
            random.shuffle(buckets[k])
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        iter_list = []
        total_len = 0
        batch = []
        len_list = self.data_source.len_list
        if self.max_token_count:
            for k in self.buckets.keys():
                for index in self.buckets[k]:
                    if total_len + len_list[index] > self.max_token_count:
                        iter_list.append(batch)
                        batch = [index]
                        total_len = len_list[index]
                    else:
                        batch.append(index)
                        total_len += len_list[index]
        else:
            for index in self.buckets[k]:
                if len(batch) == self.batch_size:
                    iter_list.append(batch)
                    batch = [index]
                    total_len = len_list[index]
                else:
                    batch.append(index)
                    total_len += len_list[index]

        for batch in iter_list:
            yield batch

    def __len__(self):
        return len(self.data_source)


class HuBERTDataSet(Dataset):
    """Create a Dataset for HuBERT model training and fine-tuning.

    Args:
        tsv_dir (str or Path): The root directory of the ``.tsv`` file list.
        phase_type (str): The type of the dataset. Options: [``train``, ``valid``, ``test``].
    """
    def __init__(
        self,
        exp_dir: Union[str, Path],
        dataset: str,
        phase_type: str,
        min_sample: int = 32000,
        max_sample: int = 250000,
    ) -> None:
        self.exp_dir = Path(exp_dir)
        tsv_dir = self.exp_dir / "tsv"
        label_dir = self.exp_dir / "label"
        f_list, ind_list, len_list = self._get_lists(
            tsv_dir,
            dataset,
            phase_type,
            min_sample,
            max_sample
        )
        self.f_list, self.ind_list, self.len_list = f_list, ind_list, len_list
        self.labels = self._load_label(label_dir, dataset, phase_type)

    def __len__(self):
        return len(self.f_list)

    def _get_lists(
        self,
        tsv_dir: Path,
        dataset: str,
        phase_type: str,
        min_sample: int,
        max_sample: int,
    ) -> Tuple[List[Path], List[int], List[int]]:
        """Get the list of paths for iteration.
        Args:
            tsv_dir (Path): The root directory of the ``.tsv`` file list.
            dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
            phase_type (str): The type of the dataset. Options: [``train``, ``valid``].
            min_sample (int): The minimum number of audio samples in the dataset.
            max_sample (int): The maximum number of audio samples in the dataset.

        Returns:
            (list) List of file paths.
            (list) List of indices that qualify ``min_sample`` <= length <= ``max_sample``.
            (list) List of waveform lengths.
        """
        f_ind_len_list = []
        with open(tsv_dir / f"{dataset}_{phase_type}.tsv") as f:
            root = f.readline().rstrip()
            for index, line in enumerate(f):
                path, nsample = line.split("\t")
                path = f"{root}/{path}"
                nsample = int(nsample)
                if min_sample <= nsample <= max_sample:
                    f_ind_len_list.append((path, index, nsample))
        f_ind_len_list.sort(key=lambda x: x[2])  # sort the file lists by the sequence length
        f_list, ind_list, len_list = [], [], []
        for ele in f_ind_len_list:
            f_list.append(ele[0])
            ind_list.append(ele[1])
            len_list.append(ele[2])
        return f_list, ind_list, len_list

    def _load_audio(
        self,
        index: int
    ) -> Tensor:
        wav_path = self.f_list[index]
        waveform, sample_rate = torchaudio.load(wav_path)
        assert waveform.shape[1] == self.len_list[index]
        return waveform

    def _load_label(
        self,
        label_dir: Path,
        dataset: str,
        phase_type: str
    ) -> List[List[int]]:
        with open(label_dir / f"{dataset}_{phase_type}.pt") as f:
            labels = [line.rstrip() for line in f]
            labels = [[int(ele) for ele in labels[i].split()] for i in self.ind_list]
        return labels

    def __getitem__(self, index):
        waveform = self._load_audio(index)
        length = waveform.shape[1]
        label = torch.Tensor(self.labels[index])
        return (waveform, length, label)


def collate_fn_hubert(
    batch
) -> Tuple[Tensor, Tensor, Tensor]:
    waveforms, lengths, labels = [], [], []
    for sample in batch:
        waveform, length, label = sample
        waveforms.append(waveform)
        lengths.append(length)
        labels.append(label)

    data = torch.zeros(len(batch), max(lengths))
    for i in range(len(waveforms)):
        data[i][0:waveforms[i].shape[1]] = waveforms[i][0]
    lengths = torch.Tensor(lengths)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return data, lengths, labels
