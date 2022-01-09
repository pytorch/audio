from parameterized import parameterized
from torchaudio.prototype.datasets import BucketizeBatchSampler
from torchaudio_unittest.common_utils import TorchaudioTestCase


class TestBucketizeBatchSampler(TorchaudioTestCase):
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
        sampler = list(BucketizeBatchSampler(lengths, num_buckets=100, batch_size=batch_size))
        for indices in sampler:
            self.assertEqual(len(indices), batch_size)

    @parameterized.expand(
        [
            (1000,),
            (1500,),
            (2000,),
        ]
    )
    def test_max_token(self, max_token_count):
        lengths = list(range(1000))
        sampler = list(BucketizeBatchSampler(lengths, num_buckets=100, max_token_count=max_token_count))

        for indices in sampler:
            self.assertLessEqual(sum([lengths[index] for index in indices]), max_token_count)

    @parameterized.expand(
        [
            (200,),
            (300,),
            (500,),
        ]
    )
    def test_max_token_fail(self, max_token_count):
        lengths = list(range(1000))
        with self.assertRaises(ValueError):
            sampler = list(BucketizeBatchSampler(lengths, num_buckets=100, max_token_count=max_token_count))
            for indices in sampler:
                self.assertLessEqual(sum([lengths[index] for index in indices]), max_token_count)

    @parameterized.expand(
        [
            [3, 0, 100],
            [4, 20, 300],
            [5, 400, 500],
        ]
    )
    def test_sampler_length(self, batch_size, min_len, max_len):
        lengths = list(range(1000))
        sampler = list(
            BucketizeBatchSampler(lengths, num_buckets=100, batch_size=batch_size, min_len=min_len, max_len=max_len)
        )

        lengths_filtered = [e for e in lengths if min_len <= e <= max_len]
        self.assertEqual(len(sampler), len(lengths_filtered) // batch_size)
