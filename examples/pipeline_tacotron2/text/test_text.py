import unittest

from parameterized import parameterized

from .text_preprocessing import text_to_sequence


class TestTextPreprocessor(unittest.TestCase):

    @parameterized.expand(
        [
            ["dr.  Strange?", [15, 26, 14, 31, 26, 29, 11, 30, 31, 29, 12, 25, 18, 16, 10]],
            ["ML, is        fun.", [24, 23, 6, 11, 20, 30, 11, 17, 32, 25, 7]],
            ["I love torchaudio!", [20, 11, 23, 26, 33, 16, 11, 31, 26, 29, 14, 19, 12, 32, 15, 20, 26, 2]],
            # 'one thousand dollars, twenty cents'
            ["$1,000.20", [26, 25, 16, 11, 31, 19, 26, 32, 30, 12, 25, 15, 11, 15, 26, 23, 23,
                           12, 29, 30, 6, 11, 31, 34, 16, 25, 31, 36, 11, 14, 16, 25, 31, 30]],
        ]
    )
    def test_text_to_sequence(self, sent, seq):

        assert (text_to_sequence(sent) == seq)
