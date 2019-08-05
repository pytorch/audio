from __future__ import absolute_import, division, print_function, unicode_literals
import os

import torch
import torchaudio
import unittest
import common_utils
import torchaudio.datasets.vctk as vctk


class TestVCTK(unittest.TestCase):
    def setUp(self):
        self.test_dirpath, self.test_dir = common_utils.create_temp_assets_dir()

    def get_full_path(self, file):
        return os.path.join(self.test_dirpath, 'assets', file)

    def test_is_audio_file(self):
        self.assertTrue(vctk.is_audio_file('foo.wav'))
        self.assertTrue(vctk.is_audio_file('foo.WAV'))
        self.assertFalse(vctk.is_audio_file('foo.bar'))

    def test_make_manifest(self):
        audios = vctk.make_manifest(self.test_dirpath)
        files = ['kaldi_file.wav', 'kaldi_file_8000.wav',
                 'sinewave.wav', 'steam-train-whistle-daniel_simon.mp3']
        files = [self.get_full_path(file) for file in files]

        audios.sort()
        self.assertEqual(files, audios, msg='files %s did not match audios %s' % (files, audios))

    def test_read_audio_downsample_false(self):
        file = self.get_full_path('kaldi_file.wav')
        s, sr = vctk.read_audio(file, downsample=False)
        self.assertEqual(sr, 16000, msg='incorrect sample rate %d' % (sr))
        self.assertEqual(s.shape, (1, 20), msg='incorrect shape %s' % (str(s.shape)))

    def test_read_audio_downsample_true(self):
        file = self.get_full_path('kaldi_file.wav')
        s, sr = vctk.read_audio(file, downsample=True)
        self.assertEqual(sr, 16000, msg='incorrect sample rate %d' % (sr))
        self.assertEqual(s.shape, (1, 20), msg='incorrect shape %s' % (str(s.shape)))

    def test_load_txts(self):
        utterences = vctk.load_txts(self.test_dirpath)
        expected_utterances = {'file2': 'word5 word6\n', 'file1': 'word1 word2\n'}
        self.assertEqual(utterences, expected_utterances,
                         msg='%s did not match %s' % (utterences, expected_utterances))

    def test_vctk(self):
        # TODO somehow test download=True, the dataset is too big download ~10 GB for
        # each test so need a way to mock it
        self.assertRaises(RuntimeError, vctk.VCTK, self.test_dirpath, download=False)

if __name__ == '__main__':
    unittest.main()
