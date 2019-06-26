import os
import random
import subprocess
import torchaudio

# Path to the compute-spectrogram-feats executable.
EXE_PATH = '/scratch/jamarshon/kaldi/src/featbin/compute-spectrogram-feats'

# Path to the scp file. An example of its contents would be "my_id /scratch/jamarshon/audio/test/assets/kaldi_file.wav"
# where the space separates an id from a wav file.
SCP_PATH = 'scp:/scratch/jamarshon/downloads/a.scp'
# The directory to which the stft features will be written to.
OUTPUT_DIR = 'ark:/scratch/jamarshon/audio/test/assets/kaldi/'

# The number of samples inside the input wave file read from `SCP_PATH`
WAV_LEN = 20

# How many output files should be generated.
NUMBER_OF_OUTPUTS = 100

VERBOSE = False


def generate_rand_boolean():
    # Generates a random boolean ('true', 'false')
    return 'true' if random.randint(0, 1) else 'false'


def generate_rand_window_type():
    # Generates a random window type
    return torchaudio.compliance.kaldi.WINDOWS[random.randint(0, len(torchaudio.compliance.kaldi.WINDOWS) - 1)]


def run():
    for i in range(NUMBER_OF_OUTPUTS):
        inputs = {
            'blackman_coeff': '%.4f' % (random.random() * 5),
            'dither': '0',
            'energy_floor': '%.4f' % (random.random() * 5),
            'frame_length': '%.4f' % (float(random.randint(2, WAV_LEN - 1)) / 16000 * 1000),
            'frame_shift': '%.4f' % (float(random.randint(1, WAV_LEN - 1)) / 16000 * 1000),
            'preemphasis_coefficient': '%.2f' % random.random(),
            'raw_energy': generate_rand_boolean(),
            'remove_dc_offset': generate_rand_boolean(),
            'round_to_power_of_two': generate_rand_boolean(),
            'snip_edges': generate_rand_boolean(),
            'subtract_mean': generate_rand_boolean(),
            'window_type': generate_rand_window_type()
        }

        fn = 'spec-' + ('-'.join(list(inputs.values())))

        out_fn = OUTPUT_DIR + fn + '.ark'

        arg = [EXE_PATH]
        arg += ['--' + k.replace('_', '-') + '=' + inputs[k] for k in inputs]
        arg += [SCP_PATH, out_fn]

        print(fn)
        print(inputs)
        print(' '.join(arg))

        try:
            if VERBOSE:
                subprocess.call(arg)
            else:
                subprocess.call(arg, stderr=open(os.devnull, 'wb'), stdout=open(os.devnull, 'wb'))
            print('success')
        except Exception:
            if os.path.exists(out_fn):
                os.remove(out_fn)


if __name__ == '__main__':
    run()
