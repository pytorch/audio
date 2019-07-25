import torchaudio


# TODO See https://github.com/pytorch/audio/issues/165
class Spectrogram:
    forward = torchaudio.transforms.Spectrogram().forward


class SpectrogramToDB:
    forward = torchaudio.transforms.SpectrogramToDB().forward


class MelScale:
    forward = torchaudio.transforms.MelScale().forward


class MelSpectrogram:
    forward = torchaudio.transforms.MelSpectrogram().forward


class MFCC:
    forward = torchaudio.transforms.MFCC().forward


class MuLawEncoding:
    forward = torchaudio.transforms.MuLawEncoding().forward


class MuLawDecoding:
    forward = torchaudio.transforms.MuLawDecoding().forward


class Resample:
    # Resample isn't a script_method
    forward = torchaudio.transforms.Resample.forward
