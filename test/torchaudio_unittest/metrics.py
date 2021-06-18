from parameterized import parameterized
from torchaudio.functional import character_edit_distance, word_edit_distance
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
        ]
    )
    def test_simple_case_character_edit_distance(self, ref, hyp, distance):
        assert character_edit_distance(ref, hyp) == distance

    @parameterized.expand(
        [
            [["hello", "world"], ["hello", "world", "!"], 1],
            [["hello", "world"], ["world", "hello", "!"], 2],
            [["hello", "world"], "world", 5],
        ]
    )
    def test_simple_case_word_edit_distance(self, ref, hyp, distance):
        assert word_edit_distance(ref, hyp) == distance