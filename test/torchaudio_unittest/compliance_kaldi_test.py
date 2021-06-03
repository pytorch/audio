import os
import math

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from torchaudio_unittest import common_utils
from .compliance import utils as compliance_utils


def extract_window(window, wave, f, frame_length, frame_shift, snip_edges):
    # just a copy of ExtractWindow from feature-window.cc in python
    def first_sample_of_frame(frame, window_size, window_shift, snip_edges):
        if snip_edges:
            return frame * window_shift
        else:
            midpoint_of_frame = frame * window_shift + window_shift // 2
            beginning_of_frame = midpoint_of_frame - window_size // 2
            return beginning_of_frame

    sample_offset = 0
    num_samples = sample_offset + wave.size(0)
    start_sample = first_sample_of_frame(f, frame_length, frame_shift, snip_edges)
    end_sample = start_sample + frame_length

    if snip_edges:
        assert(start_sample >= sample_offset and end_sample <= num_samples)
    else:
        assert(sample_offset == 0 or start_sample >= sample_offset)

    wave_start = start_sample - sample_offset
    wave_end = wave_start + frame_length
    if wave_start >= 0 and wave_end <= wave.size(0):
        window[f, :] = wave[wave_start:(wave_start + frame_length)]
    else:
        wave_dim = wave.size(0)
        for s in range(frame_length):
            s_in_wave = s + wave_start
            while s_in_wave < 0 or s_in_wave >= wave_dim:
                if s_in_wave < 0:
                    s_in_wave = - s_in_wave - 1
                else:
                    s_in_wave = 2 * wave_dim - 1 - s_in_wave
            window[f, s] = wave[s_in_wave]


@common_utils.skipIfNoSox
class Test_Kaldi(common_utils.TempDirMixin, common_utils.TorchaudioTestCase):

    kaldi_output_dir = common_utils.get_asset_path('kaldi')
    test_filepath = common_utils.get_asset_path('kaldi_file.wav')
    test_filepaths = {prefix: [] for prefix in compliance_utils.TEST_PREFIX}

    # separating test files by their types (e.g 'spec', 'fbank', etc.)
    for f in os.listdir(kaldi_output_dir):
        dash_idx = f.find('-')
        assert f.endswith('.ark') and dash_idx != -1
        key = f[:dash_idx]
        assert key in test_filepaths
        test_filepaths[key].append(f)

    def _test_get_strided_helper(self, num_samples, window_size, window_shift, snip_edges):
        waveform = torch.arange(num_samples).float()
        output = kaldi._get_strided(waveform, window_size, window_shift, snip_edges)

        # from NumFrames in feature-window.cc
        n = window_size
        if snip_edges:
            m = 0 if num_samples < window_size else 1 + (num_samples - window_size) // window_shift
        else:
            m = (num_samples + (window_shift // 2)) // window_shift

        self.assertTrue(output.dim() == 2)
        self.assertTrue(output.shape[0] == m and output.shape[1] == n)

        window = torch.empty((m, window_size))

        for r in range(m):
            extract_window(window, waveform, r, window_size, window_shift, snip_edges)
        self.assertEqual(window, output)

    def test_get_strided(self):
        # generate any combination where 0 < window_size <= num_samples and
        # 0 < window_shift.
        for num_samples in range(1, 20):
            for window_size in range(1, num_samples + 1):
                for window_shift in range(1, 2 * num_samples + 1):
                    for snip_edges in range(0, 2):
                        self._test_get_strided_helper(num_samples, window_size, window_shift, snip_edges)

    def _create_data_set(self):
        # used to generate the dataset to test on. this is not used in testing (offline procedure)
        sr = 16000
        x = torch.arange(0, 20).float()
        # between [-6,6]
        y = torch.cos(2 * math.pi * x) + 3 * torch.sin(math.pi * x) + 2 * torch.cos(x)
        # between [-2^30, 2^30]
        y = (y / 6 * (1 << 30)).long()
        # clear the last 16 bits because they aren't used anyways
        y = ((y >> 16) << 16).float()
        torchaudio.save(self.test_filepath, y, sr)
        sound, sample_rate = common_utils.load_wav(self.test_filepath, normalize=False)
        print(y >> 16)
        self.assertTrue(sample_rate == sr)
        self.assertEqual(y, sound)

    def _print_diagnostic(self, output, expect_output):
        # given an output and expected output, it will print the absolute/relative errors (max and mean squared)
        abs_error = output - expect_output
        abs_mse = abs_error.pow(2).sum() / output.numel()
        abs_max_error = torch.max(abs_error.abs())

        relative_error = abs_error / expect_output
        relative_mse = relative_error.pow(2).sum() / output.numel()
        relative_max_error = torch.max(relative_error.abs())

        print('abs_mse:', abs_mse.item(), 'abs_max_error:', abs_max_error.item())
        print('relative_mse:', relative_mse.item(), 'relative_max_error:', relative_max_error.item())

    def _compliance_test_helper(self, sound_filepath, filepath_key, expected_num_files,
                                expected_num_args, get_output_fn, atol=1e-5, rtol=1e-7):
        """
        Inputs:
            sound_filepath (str): The location of the sound file
            filepath_key (str): A key to `test_filepaths` which matches which files to use
            expected_num_files (int): The expected number of kaldi files to read
            expected_num_args (int): The expected number of arguments used in a kaldi configuration
            get_output_fn (Callable[[Tensor, List], Tensor]): A function that takes in a sound signal
                and a configuration and returns an output
            atol (float): absolute tolerance
            rtol (float): relative tolerance
        """
        sound, sr = common_utils.load_wav(sound_filepath, normalize=False)
        files = self.test_filepaths[filepath_key]

        assert len(files) == expected_num_files, \
            ('number of kaldi {} file changed to {}'.format(
                filepath_key, len(files)))

        for f in files:
            print(f)

            # Read kaldi's output from file
            kaldi_output_path = os.path.join(self.kaldi_output_dir, f)
            kaldi_output_dict = dict(torchaudio.kaldi_io.read_mat_ark(kaldi_output_path))

            assert len(kaldi_output_dict) == 1 and 'my_id' in kaldi_output_dict, 'invalid test kaldi ark file'
            kaldi_output = kaldi_output_dict['my_id']

            # Construct the same configuration used by kaldi
            args = f.split('-')
            args[-1] = os.path.splitext(args[-1])[0]
            assert len(args) == expected_num_args, 'invalid test kaldi file name'
            args = [compliance_utils.parse(arg) for arg in args]

            output = get_output_fn(sound, args)

            self._print_diagnostic(output, kaldi_output)
            self.assertEqual(output, kaldi_output, atol=atol, rtol=rtol)

    def test_mfcc_empty(self):
        # Passing in an empty tensor should result in an error
        self.assertRaises(AssertionError, kaldi.mfcc, torch.empty(0))
