try:
    from .fb import download_url_to_file, load_state_dict_from_url
except ImportError:
    from torch.hub import download_url_to_file, load_state_dict_from_url


__all__ = [
    "load_state_dict_from_url",
    "download_url_to_file",
]
