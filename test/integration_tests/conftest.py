import os
import shutil

import pytest
import torch
import torchaudio


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
        return "".join(hypothesis)


@pytest.fixture
def ctc_decoder():
    return GreedyCTCDecoder


_FILES = {
    "en": "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.flac",
    "de": "20090505-0900-PLENARY-16-de_20090505-21_56_00_8.flac",
    "en2": "20120613-0900-PLENARY-8-en_20120613-13_46_50_3.flac",
    "es": "20130207-0900-PLENARY-7-es_20130207-13_02_05_5.flac",
    "fr": "20121212-0900-PLENARY-5-fr_20121212-11_37_04_10.flac",
    "it": "20170516-0900-PLENARY-16-it_20170516-18_56_31_1.flac",
}
_MIXTURE_FILES = {
    "speech_separation": "mixture_3729-6852-0037_8463-287645-0000.wav",
    "music_separation": "al_james_mixture_shorter.wav",
}
_CLEAN_FILES = {
    "speech_separation": [
        "s1_3729-6852-0037_8463-287645-0000.wav",
        "s2_3729-6852-0037_8463-287645-0000.wav",
    ],
    "music_separation": [
        "al_james_drums_shorter.wav",
        "al_james_bass_shorter.wav",
        "al_james_other_shorter.wav",
        "al_james_vocals_shorter.wav",
    ],
}


@pytest.fixture
def sample_speech(lang):
    if lang not in _FILES:
        raise NotImplementedError(f"Unexpected lang: {lang}")
    filename = _FILES[lang]
    path = torchaudio.utils.download_asset(f"test-assets/{filename}")
    return path


@pytest.fixture
def mixture_source(task):
    if task not in _MIXTURE_FILES:
        raise NotImplementedError(f"Unexpected task: {task}")
    path = torchaudio.utils.download_asset(f"test-assets/{_MIXTURE_FILES[task]}")
    return path


@pytest.fixture
def clean_sources(task):
    if task not in _CLEAN_FILES:
        raise NotImplementedError(f"Unexpected task: {task}")
    paths = []
    for file in _CLEAN_FILES[task]:
        path = torchaudio.utils.download_asset(f"test-assets/{file}")
        paths.append(path)
    return paths


def pytest_addoption(parser):
    parser.addoption(
        "--use-tmp-hub-dir",
        action="store_true",
        help=(
            "When provided, tests will use temporary directory as Torch Hub directory. "
            "Downloaded models will be deleted after each test."
        ),
    )


@pytest.fixture(autouse=True)
def temp_hub_dir(tmp_path, pytestconfig):
    if not pytestconfig.getoption("use_tmp_hub_dir"):
        yield
    else:
        org_dir = torch.hub.get_dir()
        subdir = os.path.join(tmp_path, "hub")
        torch.hub.set_dir(subdir)
        yield
        torch.hub.set_dir(org_dir)
        shutil.rmtree(subdir, ignore_errors=True)


@pytest.fixture()
def emissions():
    path = torchaudio.utils.download_asset("test-assets/emissions-8555-28447-0012.pt")
    return torch.load(path)
