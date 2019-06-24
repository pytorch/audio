import math
import os
import test.common_utils
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import unittest


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


class Test_Kaldi(unittest.TestCase):
    test_dirpath, test_dir = test.common_utils.create_temp_assets_dir()
    test_filepath = os.path.join(test_dirpath, 'assets', 'kaldi_file.wav')
    kaldi_output_dir = os.path.join(test_dirpath, 'assets', 'kaldi')
    test_filepaths = {'spec': [], 'fbank': []}

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
        self.assertTrue(torch.allclose(window, output))

    def a_test_get_strided(self):
        # generate any combination where 0 < window_size <= num_samples and
        # 0 < window_shift.
        for num_samples in range(1, 20):
            for window_size in range(1, num_samples + 1):
                for window_shift in range(1, 2 * num_samples + 1):
                    for snip_edges in range(0, 2):
                        self._test_get_strided_helper(num_samples, window_size, window_shift, snip_edges)

    def _create_data_set(self):
        # used to generate the dataset to test on. this is not used in testing (offline procedure)
        test_dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_filepath = os.path.join(test_dirpath, 'assets', 'kaldi_file.wav')
        sr = 16000
        x = torch.arange(0, 20).float()
        # between [-6,6]
        y = torch.cos(2 * math.pi * x) + 3 * torch.sin(math.pi * x) + 2 * torch.cos(x)
        # between [-2^30, 2^30]
        y = (y / 6 * (1 << 30)).long()
        # clear the last 16 bits because they aren't used anyways
        y = ((y >> 16) << 16).float()
        torchaudio.save(test_filepath, y, sr)
        sound, sample_rate = torchaudio.load(test_filepath, normalization=False)
        print(y >> 16)
        self.assertTrue(sample_rate == sr)
        self.assertTrue(torch.allclose(y, sound))

    def a_test_spectrogram(self):
        sound, sample_rate = torchaudio.load_wav(self.test_filepath)
        files = self.test_filepaths['spec']

        assert len(files) == 131, ('number of kaldi spec file changed to %d' % (len(files)))

        for f in files:
            print(f)
            kaldi_output_path = os.path.join(self.kaldi_output_dir, f)
            kaldi_output_dict = {k: v for k, v in torchaudio.kaldi_io.read_mat_ark(kaldi_output_path)}

            assert len(kaldi_output_dict) == 1 and 'my_id' in kaldi_output_dict, 'invalid test kaldi ark file'
            kaldi_output = kaldi_output_dict['my_id']

            args = f.split('-')
            args[-1] = os.path.splitext(args[-1])[0]
            assert len(args) == 13, 'invalid test kaldi file name'

            spec_output = kaldi.spectrogram(
                sound,
                blackman_coeff=float(args[1]),
                dither=float(args[2]),
                energy_floor=float(args[3]),
                frame_length=float(args[4]),
                frame_shift=float(args[5]),
                preemphasis_coefficient=float(args[6]),
                raw_energy=args[7] == 'true',
                remove_dc_offset=args[8] == 'true',
                round_to_power_of_two=args[9] == 'true',
                snip_edges=args[10] == 'true',
                subtract_mean=args[11] == 'true',
                window_type=args[12])

            error = spec_output - kaldi_output
            mse = error.pow(2).sum() / spec_output.numel()
            max_error = torch.max(error.abs())

            print('mse:', mse.item(), 'max_error:', max_error.item())
            self.assertTrue(spec_output.shape, kaldi_output.shape)
            self.assertTrue(torch.allclose(spec_output, kaldi_output, atol=1e-3, rtol=0))

    def test_fbank(self):
        sound, sample_rate = torchaudio.load_wav(self.test_filepath)
        files = self.test_filepaths['fbank']

        assert len(files) == 49, ('number of kaldi spec file changed to %d' % (len(files)))

        for f in files:
            print(f)
            kaldi_output_path = os.path.join(self.kaldi_output_dir, f)
            kaldi_output_dict = {k: v for k, v in torchaudio.kaldi_io.read_mat_ark(kaldi_output_path)}

            if not (len(kaldi_output_dict) == 1 and 'my_id' in kaldi_output_dict):
                os.remove('/scratch/jamarshon/audio/test/assets/kaldi/' + f)
                continue
            assert len(kaldi_output_dict) == 1 and 'my_id' in kaldi_output_dict, 'invalid test kaldi ark file'
            kaldi_output = kaldi_output_dict['my_id']

            args = f.split('-')
            args[-1] = os.path.splitext(args[-1])[0]
            assert len(args) == 22, 'invalid test kaldi file name'

            spec_output = kaldi.fbank(
                sound,
                blackman_coeff=float(args[1]),
                dither=0.0,
                energy_floor=float(args[2]),
                frame_length=float(args[3]),
                frame_shift=float(args[4]),
                high_freq= float(args[5]),
                htk_compat=args[6] == 'true',
                low_freq=float(args[7]),
                num_mel_bins=int(args[8]),
                preemphasis_coefficient=float(args[9]),
                raw_energy=args[10] == 'true',
                remove_dc_offset=args[11] == 'true',
                round_to_power_of_two=args[12] == 'true',
                snip_edges=args[13] == 'true',
                subtract_mean=args[14] == 'true',
                use_energy=args[15] == 'true',
                use_log_fbank=args[16] == 'true',
                use_power=args[17] == 'true',
                vtln_high=float(args[18]),
                vtln_low=float(args[19]),
                vtln_warp=float(args[20]),
                window_type=args[21])

            error = spec_output - kaldi_output
            mse = error.pow(2).sum() / spec_output.numel()
            max_error = torch.max(error.abs())

            print('mse:', mse.item(), 'max_error:', max_error.item())
            # self.assertTrue(spec_output.shape, kaldi_output.shape)
            # self.assertTrue(torch.allclose(spec_output, kaldi_output, atol=1e-3, rtol=0))

if __name__ == '__main__':
    unittest.main()
