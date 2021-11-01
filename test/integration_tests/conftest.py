import torch
import pytest


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank: int = 0):
        super().__init__()
        self.blank = blank
        self.labels = labels

    def forward(self, logits: torch.Tensor) -> str:
        """Given a sequence logits over labels, get the best path string

        Args:
            logits (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        best_path = torch.argmax(logits, dim=-1)  # [num_seq,]
        best_path = torch.unique_consecutive(best_path, dim=-1)
        hypothesis = []
        for i in best_path:
            if i != self.blank:
                hypothesis.append(self.labels[i])
        return ''.join(hypothesis)


@pytest.fixture
def ctc_decoder():
    return GreedyCTCDecoder


_FILES = {
    'en': (
        'Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.flac',
        '5005909deb42b30b5f95e0d7109455300de467f8750ff8b3515b1477c31eac24',
    ),
    'es': (
        '20130207-0900-PLENARY-7-es_20130207-13_02_05_5.flac',
        '293888e3bd729b946e4b32babc0c9bb6aa9c7d190a8a52663a1d772ea8026fea',
    ),
    'fr': (
        '20121212-0900-PLENARY-5-fr_20121212-11_37_04_10.flac',
        '0ef184b9098936f1ca7f215395ba60ad9e90163482d5f49898f33b653ec64f33',
    ),
}


@pytest.fixture
def sample_speech(tmp_path, lang):
    if lang not in _FILES:
        raise NotImplementedError(f'Unexpected lang: {lang}')
    filename, hash_prefix = _FILES[lang]
    path = tmp_path.parent / filename
    if not path.exists():
        url = f'https://download.pytorch.org/torchaudio/test-assets/{filename}'
        print(f'Downloading: {url}')
        torch.hub.download_url_to_file(url, path, hash_prefix=hash_prefix, progress=False)
    return path
