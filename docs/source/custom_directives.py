import hashlib
from pathlib import Path
from typing import List
from urllib.parse import quote, urlencode

import requests
from docutils import nodes
from docutils.parsers.rst.directives.images import Image


_THIS_DIR = Path(__file__).parent

# Color palette from PyTorch Developer Day 2021 Presentation Template
YELLOW = "F9DB78"
GREEN = "70AD47"
BLUE = "00B0F0"
PINK = "FF71DA"
ORANGE = "FF8300"
TEAL = "00E5D1"
GRAY = "7F7F7F"


def _get_cache_path(key, ext):
    filename = f"{hashlib.sha256(key).hexdigest()}{ext}"
    cache_dir = _THIS_DIR / "gen_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def _download(url, path):
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as file:
        file.write(response.content)


def _fetch_image(url):
    path = _get_cache_path(url.encode("utf-8"), ext=".svg")
    if not path.exists():
        _download(url, path)
    return str(path.relative_to(_THIS_DIR))


class BaseShield(Image):
    def run(self, params, alt, section) -> List[nodes.Node]:
        url = f"https://img.shields.io/static/v1?{urlencode(params, quote_via=quote)}"
        path = _fetch_image(url)
        self.arguments = [path]
        self.options["alt"] = alt
        if "class" not in self.options:
            self.options["class"] = []
        self.options["class"].append("shield-badge")
        self.options["target"] = f"supported_features.html#{section}"
        return super().run()


def _parse_device_args(arg: str):
    devices = sorted(arg.strip().split())

    valid_values = {"CPU", "CUDA"}
    if any(val not in valid_values for val in devices):
        raise ValueError(
            f"One or more device values are not valid. The valid values are {valid_values}. Given value: '{arg}'"
        )
    return ", ".join(sorted(devices))


def _parse_feature_args(arg: str):
    features = sorted(arg.strip().split())

    valid_values = {"Autograd", "TorchScript", "Kaldi-Compatible"}
    if any(val not in valid_values for val in features):
        raise ValueError(
            "One or more feature values are not valid. "
            f"The valid values are {valid_values}. "
            f"Given value: '{arg}'"
        )
    return ", ".join(sorted(features))


class SupportedDevices(BaseShield):
    """List the supported devices"""

    required_arguments = 1
    final_argument_whitespace = True

    def run(self) -> List[nodes.Node]:
        devices = _parse_device_args(self.arguments[0])
        alt = f"This feature supports the following devices: {devices}"
        params = {
            "label": "Supported Devices",
            "message": devices,
            "labelColor": GRAY,
            "color": BLUE,
            "style": "flat-square",
        }
        return super().run(params, alt, "devices")


class SupportedFeatures(BaseShield):
    """List the supported features"""

    required_arguments = 1
    final_argument_whitespace = True

    def run(self) -> List[nodes.Node]:
        features = _parse_feature_args(self.arguments[0])
        alt = f"This feature supports the following features: {features}"
        params = {
            "label": "Supported Features",
            "message": features,
            "labelColor": GRAY,
            "color": GREEN,
            "style": "flat-square",
        }
        return super().run(params, alt, "features")
