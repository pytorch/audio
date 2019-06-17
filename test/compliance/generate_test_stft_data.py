import random

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

WINDOWS = ['hamming', 'hanning', 'povey', 'rectangular', 'blackman']


def generate_rand_boolean():
    # Generates a random boolean ('true', 'false')
    return 'true' if random.randint(0, 1) else 'false'


def generate_rand_window_type():
    # Generates a random window type
    return WINDOWS[random.randint(0, len(WINDOWS) - 1)]


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

        fn = '-'.join(list(inputs.values()))

        arg = [
            EXE_PATH,
            '--blackman-coeff=' + inputs['blackman_coeff'],
            '--dither=' + inputs['dither'],
            '--energy-floor=' + inputs['energy_floor'],
            '--frame-length=' + inputs['frame_length'],
            '--frame-shift=' + inputs['frame_shift'],
            '--preemphasis-coefficient=' + inputs['preemphasis_coefficient'],
            '--raw-energy=' + inputs['raw_energy'],
            '--remove-dc-offset=' + inputs['remove_dc_offset'],
            '--round-to-power-of-two=' + inputs['round_to_power_of_two'],
            '--sample-frequency=16000',
            '--snip-edges=' + inputs['snip_edges'],
            '--subtract-mean=' + inputs['subtract_mean'],
            '--window-type=' + inputs['window_type'],
            SCP_PATH,
            OUTPUT_DIR + fn + '.ark'
        ]

        print(fn)
        print(inputs)
        print(' '.join(arg))

        try:
            subprocess.call(arg)
        except Exception:
            pass


if __name__ == '__main__':
    run()
