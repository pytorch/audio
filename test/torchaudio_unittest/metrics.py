from parameterized import parameterized
from torchaudio.metrics import levenshtein_distance
from torchaudio_unittest import common_utils


class TestLevenshteinDistance(common_utils.TorchaudioTestCase):
    @parameterized.expand(
        [
            ["abc", "", 3],
            ["", "", 0],
            ["abc", "abc", 0],
            ["aaa", "aba", 1],
            ["aba", "aaa", 1],
            ["aa", "aaa", 1],
            ["aaa", "aa", 1],
            ["abc", "bcd", 2],
            [["hello", "world"], ["hello", "world", "!"], 1],
            [["hello", "world"], ["world", "hello", "!"], 2],
            [["hello", "world"], "world", 5],
        ]
    )
    def test_simple_case(self, ref, hyp, distance):
        assert levenshtein_distance(ref, hyp) == distance
