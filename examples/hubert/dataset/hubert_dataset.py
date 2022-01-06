import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import BatchSampler, Dataset


class BucketizeSampler(BatchSampler):
    """Buketize sampler for data with different lengths to reduce number of paddings.

    Args:
        lengths (List[int]): The lengths of the samples in the dataset.
        num_buckets (int): The number of buckets to split the data samples.
        min_len (int, optional): The minimum sample lengths to keep.
            (Default: 0)
        max_len (int or None, optional): The maximum sample lengths to keep. Inferred if not provided.
            (Default ``None``)
        max_token_count (int or None, optional): The max number of tokens in one mini-batch.
            (Default: ``None``)
        batch_size (int or None, optional): The number of samples in one mini-batch.
             (Default: ``None``)
        shuffle (bool, optional): Whether to shuffle buckets for non-monotonic length sampling.
             (Default True)

    Note: If ``max_token_count`` is not ``None``, the ``batch_size`` couldn't be set since
        the lengths of samples are unknown, the batch size may be different for different
        mini-batches.
    """

    def __init__(
        self,
        lengths: List[int],
        num_buckets: int,
        min_len: int = 0,
        max_len: Optional[int] = None,
        max_token_count: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        if max_len is None:
            max_len = max(lengths)

        if not (0 <= min_len <= max_len):
            raise AssertionError("``min_len`` should be non-negative and smaller than ``max_len``")
        if max_token_count is not None and batch_size is not None:
            raise AssertionError("The ``max_token_count`` and ``batch_size`` can't be both set.")
        # Filter out samples which are outside the bounds of [min_len, max_len]
        # sort to minimize gap when bucketizing.
        filtered_length_idx = [(length, i) for i, length in enumerate(lengths) if min_len <= length <= max_len]
        if len(filtered_length_idx) == 0:
            raise AssertionError("``lengths`` cannot be empty after filtering.")
        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.max_token_count = max_token_count
        self.batch_size = batch_size
        self.buckets = self._get_buckets(self.lengths, self.indices, num_buckets, min_len, max_len)
        self.shuffle = shuffle

    def _get_buckets(
        self, lengths: List[int], indices: List[int], num_buckets: int, min_len: int, max_len: int
    ) -> Dict[int, Tensor]:
        """Generate buckets based on the dataset.
        Args:
            lengths (List[int]): The lengths of the samples in the dataset.
            indices (List[int]): The indices of the samples in the original dataset.
            num_buckets (int): The number of buckets.
            min_len (int): The lower bound of the evenly spaced length intervals to determine bucket width.
            max_len (int): The upper bound of the evenly spaced length intervals to determine bucket width.

        Returns:
            (dict[int, Tensor]): A dictionary in which the key is the bucket index, the value is
                the Tensor of corresponding sample indices.
        """
        buckets = {}

        boundaries = [min_len - 1]
        interval = (max_len - min_len) // num_buckets
        for i in range(1, num_buckets):
            boundaries.append(min_len + i * interval)
        boundaries.append(max_len + 1)
        bucket_ids = torch.bucketize(torch.tensor(lengths), torch.tensor(boundaries))
        for i in indices:
            bucket_id = bucket_ids[i]
            if bucket_id in buckets:
                buckets[bucket_id].append(i)
            else:
                buckets[bucket_id] = [i]
        for k in buckets:
            if self.shuffle:
                random.shuffle(buckets[k])
            buckets[k] = torch.as_tensor(buckets[k], dtype=torch.int)
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        iter_list = []
        total_len = 0
        batch = []
        if self.max_token_count:
            for k in self.buckets.keys():
                for i in range(self.buckets[k].size(0)):
                    index = self.buckets[k][i]
                    if total_len > self.max_token_count:
                        iter_list.append(batch)
                        batch = [index]
                        total_len = self.lengths[index]
                    else:
                        batch.append(index)
                        total_len += self.lengths[index]
        else:
            for k in self.buckets.keys():
                for i in range(self.buckets[k].size(0)):
                    index = self.buckets[k][i]
                    if total_len == self.batch_size:
                        iter_list.append(batch)
                        batch = [index]
                        total_len = 1
                    else:
                        batch.append(index)
                        total_len += 1

        for batch in iter_list:
            yield batch

    def __len__(self):
        return len(self.data_source)


class HuBERTDataSet(Dataset):
    """Create a Dataset for HuBERT model training and fine-tuning.

    Args:
        exp_dir (str or Path): The root directory of the ``.tsv`` file list.
        dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
        subset (str): The subset of the dataset. Options: [``train``, ``valid``].
        min_sample (int): The minimum number of audio samples in the dataset. (Default: 32000)
        max_sample (int): The maximum number of audio samples in the dataset. (Default: 250000)
    """

    def __init__(
        self,
        exp_dir: Union[str, Path],
        dataset: str,
        subset: str,
        min_sample: int = 32000,
        max_sample: int = 250000,
    ) -> None:
        self.exp_dir = Path(exp_dir)
        tsv_dir = self.exp_dir / "tsv"
        label_dir = self.exp_dir / "label"
        f_list, ind_list, len_list = self._get_lists(tsv_dir, dataset, subset, min_sample, max_sample)
        self.f_list, self.ind_list, self.len_list = f_list, ind_list, len_list
        self.labels = self._load_labels(label_dir, dataset, subset)

    def __len__(self):
        return len(self.f_list)

    def _get_lists(
        self,
        tsv_dir: Path,
        dataset: str,
        subset: str,
        min_sample: int,
        max_sample: int,
    ) -> Tuple[List[Path], List[int], List[int]]:
        """Get the list of paths for iteration.
        Args:
            tsv_dir (Path): The root directory of the ``.tsv`` file list.
            dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
            subset (str): The subset of the dataset. Options: [``train``, ``valid``].
            min_sample (int): The minimum number of audio samples in the dataset.
            max_sample (int): The maximum number of audio samples in the dataset.

        Returns:
            (numpy.array) List of file paths.
            (numpy.array) List of indices that qualify ``min_sample`` <= length <= ``max_sample``.
            (numpy.array) List of waveform lengths.
        """
        f_ind_len_list = []
        with open(tsv_dir / f"{dataset}_{subset}.tsv") as f:
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
        return np.asarray(f_list), np.asarray(ind_list), np.asarray(len_list)

    def _load_audio(self, index: int) -> Tensor:
        """Load waveform given the sample index of the dataset.
        Args:
            index (int): The sample index.

        Returns:
            (Tensor): The corresponding waveform Tensor.
        """
        wav_path = self.f_list[index]
        waveform, sample_rate = torchaudio.load(wav_path)
        assert waveform.shape[1] == self.len_list[index]
        return waveform

    def _load_labels(self, label_dir: Path, dataset: str, subset: str) -> np.array:
        """Load all labels to memory into a numpy array.
        Args:
            label_dir (Path): The directory that contains the label file.
            dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
            subset (str): The subset of the dataset. Options: [``train``, ``valid``].

        Returns:
            (np.array): The numpy arrary that contains the labels for each audio file.
        """
        with open(label_dir / f"{dataset}_{subset}.pt") as f:
            labels = [line.rstrip() for line in f]
            labels = [labels[i] for i in self.ind_list]
        return np.asarray(labels, dtype=np.string_)

    def __getitem__(self, index):
        waveform = self._load_audio(index)
        length = waveform.shape[1]
        label = [int(ele) for ele in self.labels[index].split()]
        label = torch.tensor(label)
        return (waveform, label, length)


class CollateFnHubert:
    """The collate class for HuBERT pre-training and fine-tuning.
    Args:
        feature_type (str): The type of features for KMeans clustering.
            Options: [``mfcc``, ``hubert``].
        pad (bool): If ``pad`` is True, the waveforms and labels will be padded
            to the max length in the mini-batch. If ``pad`` is False, the waveforms
            and labels will be cropped to the minimum length in the mini-batch.
            (Default: False)
        rand_crop (bool): if ``rand_crop`` is True, the starting index of the
            waveform and label is random if the length is longer than the minimum
            length in the mini-batch.
    """

    def __init__(
        self,
        feature_type: str,
        pad: bool = False,
        rand_crop: bool = True,
    ) -> None:
        self.feature_type = feature_type
        self.pad = pad
        self.rand_crop = rand_crop

    def __call__(self, batch: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (List[Tuple(Tensor, Tensor, int)]):
                The list of tuples that contains the waveforms, labels, and audio lengths.

        Returns:
            (Tuple(Tensor, Tensor, Tensor)):
                The Tensor of waveforms of dimension `[batch, time]`.
                The Tensor of labels of dimension `[batch, seq]`.
                The Tensor of audio lengths of dimension `[batch,]`.
        """
        audio_sizes = [sample[0].shape[1] for sample in batch]
        if self.pad:
            audio_size = max(audio_sizes)
        else:
            audio_size = min(audio_sizes)
        waveforms, labels, lengths = [], [], []
        for sample in batch:
            waveform, label, length = sample
            if self.feature_type == "mfcc":
                label = label[::2]
            waveform, label, length = self._collate_audio_label(waveform, label, length, audio_size, self.rand_crop)
            waveforms.append(waveform)
            lengths.append(length)
            labels.append(label)

        data = torch.zeros(len(batch), audio_size)
        for i in range(len(waveforms)):
            data[i][0 : waveforms[i].shape[1]] = waveforms[i][0]
        lengths = torch.tensor(lengths)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return data, labels, lengths

    def _collate_audio_label(
        self,
        waveform: Tensor,
        label: Tensor,
        length: Tensor,
        audio_size: int,
        rand_crop: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Collate the audio and label at the same time.
        Args:
            waveform (Tensor): The waveform Tensor of dimension `[1, time]`.
            label (Tensor): The label Tensor of dimension `[1, seq]`.
            length (Tensor): The length Tensor of dimension `[1,]`.
            audio_size (int): The final length of the waveform.
            rand_crop (bool): if ``rand_crop`` is True, the starting index of the
                waveform and label is random if the length is longer than the minimum
                length in the mini-batch.

        Returns:
            (Tuple(Tensor, Tensor, Tensor)): Returns the Tensors for the waveform,
                label, and the waveform length.
        """
        kernel_size = 25
        stride = 20
        sample_rate = 16  # 16 per millisecond
        if waveform.shape[1] > audio_size:
            diff = waveform.size(1) - audio_size
            audio_start = torch.randint(diff, size=(1,)) if rand_crop else 0
            label_start = torch.div(
                audio_start - kernel_size * sample_rate, stride * sample_rate, rounding_mode="floor"
            )
            label_size = torch.div(audio_size - kernel_size * sample_rate, stride * sample_rate, rounding_mode="floor")
            waveform = waveform[:, audio_start : audio_start + audio_size]
            label = label[label_start : label_start + label_size]
            length = audio_size
        return waveform, label, length
