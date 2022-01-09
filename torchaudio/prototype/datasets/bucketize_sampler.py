from typing import Dict, Iterator, List, Optional

import torch
from torch import Tensor
from torch.utils.data import BatchSampler


class BucketizeBatchSampler(BatchSampler):
    """Buketized BatchSampler for sequential data with different lengths to reduce number of paddings.

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

    Note: 
        ``max_token_count`` and ``batch_size`` are mutually exclusive. Only one argument of the two
        should have value.
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
        if max_token_count is None and batch_size is None:
            raise AssertionError("One of ``max_token_count`` or ``batch_size`` must be set.")
        # Filter out samples which are outside the bounds of [min_len, max_len]
        filtered_length_idx = [(length, i) for i, length in enumerate(lengths) if min_len <= length <= max_len]
        if len(filtered_length_idx) == 0:
            raise AssertionError("``lengths`` cannot be empty after filtering.")
        # sort to minimize gap when bucketizing.
        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.index2iter = {e: i for i, e in enumerate(self.indices)}
        self.max_token_count = max_token_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buckets = self._get_buckets(self.lengths, self.indices, num_buckets, min_len, max_len)

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
            bucket_id = bucket_ids[self.index2iter[i]]
            if bucket_id in buckets:
                buckets[bucket_id].append(i)
            else:
                buckets[bucket_id] = [i]
        for k in buckets:
            buckets[k] = torch.as_tensor(buckets[k], dtype=torch.int)
            if self.shuffle:
                buckets[k] = buckets[k][torch.randperm(buckets[k].size(0))]
        return buckets

    def __iter__(self) -> Iterator[List[int]]:
        iter_list = []
        total_len = 0
        batch = []
        if self.max_token_count is not None:
            for k in self.buckets.keys():
                for i in range(self.buckets[k].size(0)):
                    index = self.buckets[k][i]
                    if len(batch) == 0 and self.lengths[self.index2iter[int(index)]] > self.max_token_count:
                        raise ValueError(
                            f"The length of the sample is {self.lengths[self.index2iter[int(index)]]}, "
                            f"which is greater than ``max_token_count``: {self.max_token_count}."
                            "Make sure you set the correct ``max_len`` and ``max_token_count``."
                        )
                    elif total_len + self.lengths[self.index2iter[int(index)]] <= self.max_token_count:
                        batch.append(int(index))
                        total_len += self.lengths[self.index2iter[int(index)]]
                    else:
                        iter_list.append(batch)
                        batch = []
                        total_len = 0

            if len(batch) > 0:
                iter_list.append(batch)
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
        if self.batch_size:
            return len(self.lengths) // self.batch_size
        else:
            return sum(self.lengths) // self.max_token_count
