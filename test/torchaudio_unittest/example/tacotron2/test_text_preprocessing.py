from parameterized import parameterized
from torchaudio._internal.module_utils import is_module_available
from torchaudio_unittest.common_utils import skipIfNoModule, TorchaudioTestCase

if is_module_available("unidecode") and is_module_available("inflect"):
    from pipeline_tacotron2.text.numbers import (
        _expand_decimal_point,
        _expand_dollars,
        _expand_number,
        _expand_ordinal,
        _expand_pounds,
        _remove_commas,
    )
    from pipeline_tacotron2.text.text_preprocessing import text_to_sequence


@skipIfNoModule("unidecode")
@skipIfNoModule("inflect")
class TestTextPreprocessor(TorchaudioTestCase):
    @parameterized.expand(
        [
            ["dr.  Strange?", [15, 26, 14, 31, 26, 29, 11, 30, 31, 29, 12, 25, 18, 16, 10]],
            ["ML, is        fun.", [24, 23, 6, 11, 20, 30, 11, 17, 32, 25, 7]],
            ["I love torchaudio!", [20, 11, 23, 26, 33, 16, 11, 31, 26, 29, 14, 19, 12, 32, 15, 20, 26, 2]],
            # 'one thousand dollars, twenty cents'
            [
                "$1,000.20",
                [
                    26,
                    25,
                    16,
                    11,
                    31,
                    19,
                    26,
                    32,
                    30,
                    12,
                    25,
                    15,
                    11,
                    15,
                    26,
                    23,
                    23,
                    12,
                    29,
                    30,
                    6,
                    11,
                    31,
                    34,
                    16,
                    25,
                    31,
                    36,
                    11,
                    14,
                    16,
                    25,
                    31,
                    30,
                ],
            ],
        ]
    )
    def test_text_to_sequence(self, sent, seq):

        assert text_to_sequence(sent) == seq

    @parameterized.expand(
        [
            ["He, she, and I have $1,000", "He, she, and I have $1000"],
        ]
    )
    def test_remove_commas(self, sent, truth):

        assert _remove_commas(sent) == truth

    @parameterized.expand(
        [
            ["He, she, and I have £1000", "He, she, and I have 1000 pounds"],
        ]
    )
    def test_expand_pounds(self, sent, truth):

        assert _expand_pounds(sent) == truth

    @parameterized.expand(
        [
            ["He, she, and I have $1000", "He, she, and I have 1000 dollars"],
            ["He, she, and I have $3000.01", "He, she, and I have 3000 dollars, 1 cent"],
            [
                "He has $500.20 and she has $1000.50.",
                "He has 500 dollars, 20 cents and she has 1000 dollars, 50 cents.",
            ],
        ]
    )
    def test_expand_dollars(self, sent, truth):

        assert _expand_dollars(sent) == truth

    @parameterized.expand(
        [
            ["1000.20", "1000 point 20"],
            ["1000.1", "1000 point 1"],
        ]
    )
    def test_expand_decimal_point(self, sent, truth):

        assert _expand_decimal_point(sent) == truth

    @parameterized.expand(
        [
            ["21st centry", "twenty-first centry"],
            ["20th centry", "twentieth centry"],
            ["2nd place.", "second place."],
        ]
    )
    def test_expand_ordinal(self, sent, truth):

        assert _expand_ordinal(sent) == truth
        _expand_ordinal,

    @parameterized.expand(
        [
            ["100020 dollars.", "one hundred thousand twenty dollars."],
            [
                "1234567890!",
                "one billion, two hundred thirty-four million, "
                "five hundred sixty-seven thousand, eight hundred ninety!",
            ],
        ]
    )
    def test_expand_number(self, sent, truth):

        assert _expand_number(sent) == truth
