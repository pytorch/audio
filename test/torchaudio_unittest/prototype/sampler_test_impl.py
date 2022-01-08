from torchaudio.prototype.datasets import BucketizeBatchSampler
from parameterized import parameterized
from torchaudio_unittest.common_utils import TestBaseMixin


class TestBucketizeBatchSampler(TestBaseMixin):
    """Test the BucketizeBatchSampler in prototype module."""

    @parameterized.expand(
        [
            (1,),
            (3,),
            (5,),
        ]
    )
    def test_batch_size(self, batch_size):
        lengths = list(range(1000))
        sampler = BucketizeBatchSampler(lengths, num_buckets=100, batch_size=batch_size)
        indices = next(sampler)
        self.assertEqual(len(indices), batch_size)

    @parameterized.expand(
        [
            [100],
            [200],
            [1000],
        ]
    )
    def test_max_token(self, max_token_count):
        lengths = list(range(1000))
        sampler = BucketizeBatchSampler(lengths, num_buckets=100, max_token_count=max_token_count)

        indices = next(sampler)
        assert sum([lengths[index] for index in indices]) <= max_token_count

    @parameterized.expand(
        [
            [3, 0, 100],
            [4, 20, 300],
            [5, 400, 500],
        ]
    )
    def test_sampler_length(self, batch_size, min_len, max_len):
        lengths = list(range(1000))
        sampler = BucketizeBatchSampler(
            lengths, num_buckets=100, batch_size=batch_size, min_len=min_len, max_len=max_len
        )

        lengths_filtered = [e for e in lengths if min_len <= e <= max_len]
        self.assertEqual(len(sampler), len(lengths_filtered) // batch_size)
