import argparse
import os
import random
import subprocess
import utils


def run(exe_path, scp_path, out_dir, wave_len, num_outputs, verbose):
    for i in range(num_outputs):
        inputs = {
            'blackman_coeff': '%.4f' % (random.random() * 5),
            'dither': '0',
            'energy_floor': '%.4f' % (random.random() * 5),
            'frame_length': '%.4f' % (float(random.randint(2, wave_len - 1)) / 16000 * 1000),
            'frame_shift': '%.4f' % (float(random.randint(1, wave_len - 1)) / 16000 * 1000),
            'preemphasis_coefficient': '%.2f' % random.random(),
            'raw_energy': utils.generate_rand_boolean(),
            'remove_dc_offset': utils.generate_rand_boolean(),
            'round_to_power_of_two': utils.generate_rand_boolean(),
            'snip_edges': utils.generate_rand_boolean(),
            'subtract_mean': utils.generate_rand_boolean(),
            'window_type': utils.generate_rand_window_type()
        }

        fn = 'spec-' + ('-'.join(list(inputs.values())))

        out_fn = out_dir + fn + '.ark'

        arg = [exe_path]
        arg += ['--' + k.replace('_', '-') + '=' + inputs[k] for k in inputs]
        arg += [scp_path, out_fn]

        print(fn)
        print(inputs)
        print(' '.join(arg))

        try:
            if verbose:
                subprocess.call(arg)
            else:
                subprocess.call(arg, stderr=open(os.devnull, 'wb'), stdout=open(os.devnull, 'wb'))
            print('success')
        except Exception:
            if os.path.exists(out_fn):
                os.remove(out_fn)


if __name__ == '__main__':
    """ Examples:
    >> python test/compliance/generate_test_stft_data.py \
        --exe_path=/scratch/jamarshon/kaldi/src/featbin/compute-spectrogram-feats \
        --scp_path=scp:/scratch/jamarshon/downloads/a.scp \
        --out_dir=ark:/scratch/jamarshon/audio/test/assets/kaldi/
    """
    parser = argparse.ArgumentParser(description='Generate spectrogram data using Kaldi.')
    parser.add_argument('--exe_path', type=str, required=True, help='Path to the compute-spectrogram-feats executable.')
    parser.add_argument('--scp_path', type=str, required=True, help='Path to the scp file. An example of its contents would be \
    "my_id /scratch/jamarshon/audio/test/assets/kaldi_file.wav". where the space separates an id from a wav file.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='The directory to which the stft features will be written to.')

    # run arguments
    parser.add_argument('--wave_len', type=int, default=20,
                        help='The number of samples inside the input wave file read from `scp_path`')
    parser.add_argument('--num_outputs', type=int, default=100, help='How many output files should be generated.')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print information.')

    args = parser.parse_args()
    run(args.exe_path, args.scp_path, args.out_dir, args.wave_len, args.num_outputs, args.verbose)
