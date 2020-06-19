import subprocess


def get_encoding(dtype):
    encodings = {
        'float32': 'floating-point',
        'int32': 'signed-integer',
        'int16': 'signed-integer',
        'uint8': 'unsigned-integer',
    }
    return encodings[dtype]


def get_bit_depth(dtype):
    bit_depths = {
        'float32': 32,
        'int32': 32,
        'int16': 16,
        'uint8': 8,
    }
    return bit_depths[dtype]


def gen_audio_file(
        path, sample_rate, num_channels,
        *, encoding=None, bit_depth=None, compression=None, attenuation=None, duration=1,
):
    """Generate synthetic audio file with `sox` command."""
    command = [
        'sox',
        '-V',  # verbose
        '--rate', str(sample_rate),
        '--null',  # no input
        '--channels', str(num_channels),
    ]
    if compression is not None:
        command += ['--compression', str(compression)]
    if bit_depth is not None:
        command += ['--bits', str(bit_depth)]
    if encoding is not None:
        command += ['--encoding', str(encoding)]
    command += [
        str(path),
        'synth', str(duration),  # synthesizes for the given duration [sec]
        'sawtooth', '1',
        # saw tooth covers the both ends of value range, which is a good property for test.
        # similar to linspace(-1., 1.)
        # this introduces bigger boundary effect than sine when converted to mp3
    ]
    if attenuation is not None:
        command += ['vol', f'-{attenuation}dB']
    print(' '.join(command))
    subprocess.run(command, check=True)
    subprocess.run(['soxi', path], check=True)
