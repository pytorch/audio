from .download import download_asset

def load_torchcodec(file, normalize=True, channels_first=True, **args):
    if not normalize:
        raise Exception("Torchcodec does not support non-normalized file reading")
    try:
        from torchcodec.decoders import AudioDecoder
    except:
         raise Exception("To use this feature, you must install torchcodec. See https://github.com/pytorch/torchcodec for installation instructions")
    decoder = AudioDecoder(file)
    if 'start_seconds' in args or 'stop_seconds' in args:
        samples = decoder.get_samples_played_in_range(**args)
    else:
        samples = decoder.get_all_samples()
    data = samples.data if channels_first else samples.data.T
    return (data, samples.sample_rate)

__all__ = [
    "load_torchcodec",
    "download_asset",
]
