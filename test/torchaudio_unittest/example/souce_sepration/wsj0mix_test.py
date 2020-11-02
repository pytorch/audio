import os

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

from source_separation.utils.dataset import wsj0mix


_FILENAMES = [
    "012c0207_1.9952_01cc0202_-1.9952.wav",
    "01co0302_1.63_014c020q_-1.63.wav",
    "01do0316_0.24011_205a0104_-0.24011.wav",
    "01lc020x_1.1301_027o030r_-1.1301.wav",
    "01mc0202_0.34056_205o0106_-0.34056.wav",
    "01nc020t_0.53821_018o030w_-0.53821.wav",
    "01po030f_2.2136_40ko031a_-2.2136.wav",
    "01ra010o_2.4098_403a010f_-2.4098.wav",
    "01xo030b_0.22377_016o031a_-0.22377.wav",
    "02ac020x_0.68566_01ec020b_-0.68566.wav",
    "20co010m_0.82801_019c0212_-0.82801.wav",
    "20da010u_1.2483_017c0211_-1.2483.wav",
    "20oo010d_1.0631_01ic020s_-1.0631.wav",
    "20sc0107_2.0222_20fo010h_-2.0222.wav",
    "20tc010f_0.051456_404a0110_-0.051456.wav",
    "407c0214_1.1712_02ca0113_-1.1712.wav",
    "40ao030w_2.4697_20vc010a_-2.4697.wav",
    "40pa0101_1.1087_40ea0107_-1.1087.wav",
]


def _mock_dataset(root_dir, num_speaker):
    dirnames = ["mix"] + [f"s{i+1}" for i in range(num_speaker)]
    for dirname in dirnames:
        os.makedirs(os.path.join(root_dir, dirname), exist_ok=True)

    seed = 0
    sample_rate = 8000
    expected = []
    for filename in _FILENAMES:
        mix = None
        src = []
        for dirname in dirnames:
            waveform = get_whitenoise(
                sample_rate=8000, duration=1, n_channels=1, dtype="int16", seed=seed
            )
            seed += 1

            path = os.path.join(root_dir, dirname, filename)
            save_wav(path, waveform, sample_rate)
            waveform = normalize_wav(waveform)

            if dirname == "mix":
                mix = waveform
            else:
                src.append(waveform)
        expected.append((sample_rate, mix, src))
    return expected


class TestWSJ0Mix2(TempDirMixin, TorchaudioTestCase):
    backend = "default"
    root_dir = None
    expected = None

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.expected = _mock_dataset(cls.root_dir, 2)

    def test_wsj0mix(self):
        dataset = wsj0mix.WSJ0Mix(self.root_dir, num_speakers=2, sample_rate=8000)

        n_ite = 0
        for i, sample in enumerate(dataset):
            (_, sample_mix, sample_src) = sample
            (_, expected_mix, expected_src) = self.expected[i]
            self.assertEqual(sample_mix, expected_mix, atol=5e-5, rtol=1e-8)
            self.assertEqual(sample_src[0], expected_src[0], atol=5e-5, rtol=1e-8)
            self.assertEqual(sample_src[1], expected_src[1], atol=5e-5, rtol=1e-8)
            n_ite += 1
        assert n_ite == len(self.expected)


class TestWSJ0Mix3(TempDirMixin, TorchaudioTestCase):
    backend = "default"
    root_dir = None
    expected = None

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.expected = _mock_dataset(cls.root_dir, 3)

    def test_wsj0mix(self):
        dataset = wsj0mix.WSJ0Mix(self.root_dir, num_speakers=3, sample_rate=8000)

        n_ite = 0
        for i, sample in enumerate(dataset):
            (_, sample_mix, sample_src) = sample
            (_, expected_mix, expected_src) = self.expected[i]
            self.assertEqual(sample_mix, expected_mix, atol=5e-5, rtol=1e-8)
            self.assertEqual(sample_src[0], expected_src[0], atol=5e-5, rtol=1e-8)
            self.assertEqual(sample_src[1], expected_src[1], atol=5e-5, rtol=1e-8)
            self.assertEqual(sample_src[2], expected_src[2], atol=5e-5, rtol=1e-8)
            n_ite += 1
        assert n_ite == len(self.expected)
