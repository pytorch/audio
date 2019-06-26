import os
import random
import subprocess
import torch
import torchaudio

# Path to the compute-fbank-feats executable.
EXE_PATH = '/scratch/jamarshon/kaldi/src/featbin/compute-fbank-feats'

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
        try:
            nyquist = 16000 // 2
            high_freq = random.randint(1, nyquist)
            low_freq = random.randint(0, high_freq - 1)
            vtln_low = random.randint(low_freq + 1, high_freq - 1)
            vtln_high = random.randint(vtln_low + 1, high_freq - 1)
            vtln_warp_factor = random.uniform(0.0, 10.0) if random.random() < 0.3 else 1.0

        except Exception:
            continue

        if not ((0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq)):
            continue
        if not (vtln_warp_factor == 1.0 or ((low_freq < vtln_low < high_freq) and
                                            (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high))):
            continue

        inputs = {
            'blackman_coeff': '%.4f' % (random.random() * 5),
            'energy_floor': '%.4f' % (random.random() * 5),
            'frame_length': '%.4f' % (float(random.randint(3, WAV_LEN - 1)) / 16000 * 1000),
            'frame_shift': '%.4f' % (float(random.randint(1, WAV_LEN - 1)) / 16000 * 1000),
            'high_freq': str(high_freq),
            'htk_compat': generate_rand_boolean(),
            'low_freq': str(low_freq),
            'num_mel_bins': str(random.randint(4, 8)),
            'preemphasis_coefficient': '%.2f' % random.random(),
            'raw_energy': generate_rand_boolean(),
            'remove_dc_offset': generate_rand_boolean(),
            'round_to_power_of_two': generate_rand_boolean(),
            'snip_edges': generate_rand_boolean(),
            'subtract_mean': generate_rand_boolean(),
            'use_energy': generate_rand_boolean(),
            'use_log_fbank': generate_rand_boolean(),
            'use_power': generate_rand_boolean(),
            'vtln_high': str(vtln_high),
            'vtln_low': str(vtln_low),
            'vtln_warp': '%.4f' % (vtln_warp_factor),
            'window_type': generate_rand_window_type()
        }

        fn = 'fbank-' + ('-'.join(list(inputs.values())))
        out_fn = OUTPUT_DIR + fn + '.ark'

        arg = [EXE_PATH]
        arg += ['--' + k.replace('_', '-') + '=' + inputs[k] for k in inputs]
        arg += ['--dither=0.0', SCP_PATH, out_fn]

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


def parse(token):
    if token == 'true':
        return True
    if token == 'false':
        return False
    if token in torchaudio.compliance.kaldi.WINDOWS or token == 'fbank':
        return token
    if '.' in token:
        return float(token)
    return int(token)


def decode(fn):
    """
    Takes a filepath and prints out the corresponding shell command to run that specific
    kaldi configuration. It also calls compliance.kaldi and prints the two outputs.

    Example:
        >> fn = 'fbank-1.1009-2.5985-1.1875-0.8750-5723-true-918-4-0.31-true-false-true-true-' \
            'false-false-false-true-4595-4281-1.0000-hamming.ark'
        >> decode(fn)
    """
    out_fn = OUTPUT_DIR + fn
    fn = fn[len('fbank-'):-len('.ark')]
    arr = [
        'blackman_coeff', 'energy_floor', 'frame_length', 'frame_shift', 'high_freq', 'htk_compat',
        'low_freq', 'num_mel_bins', 'preemphasis_coefficient', 'raw_energy', 'remove_dc_offset',
        'round_to_power_of_two', 'snip_edges', 'subtract_mean', 'use_energy', 'use_log_fbank',
        'use_power', 'vtln_high', 'vtln_low', 'vtln_warp', 'window_type']
    fn_split = fn.split('-')
    assert len(fn_split) == len(arr), ('Len mismatch: %d and %d' % (len(fn_split), len(arr)))
    inputs = {arr[i]: parse(fn_split[i]) for i in range(len(arr))}

    # print flags for C++
    s = ' '.join(['--' + arr[i].replace('_', '-') + '=' + fn_split[i] for i in range(len(arr))])
    print(EXE_PATH + ' --dither=0.0 --debug-mel=true ' + s + ' ' + SCP_PATH + ' ' + out_fn)
    print()
    # print args for python
    inputs['dither'] = 0.0
    print(inputs)
    sound, sample_rate = torchaudio.load_wav('/scratch/jamarshon/audio/test/assets/kaldi_file.wav')
    kaldi_output_dict = {k: v for k, v in torchaudio.kaldi_io.read_mat_ark(out_fn)}
    res = torchaudio.compliance.kaldi.fbank(sound, **inputs)
    torch.set_printoptions(precision=10, sci_mode=False)
    print(res)
    print(kaldi_output_dict['my_id'])


if __name__ == '__main__':
    run()
