import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_METRICS_REFERENCE = os.path.join(_THIS_DIR, "metrics", "reference.py")


sys.path.append(os.path.join(_THIS_DIR, "..", "..", "..", "test"))


def _download_metrics_reference():
    import requests

    url = "https://raw.githubusercontent.com/naplab/Conv-TasNet/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py"
    resp = requests.get(url, allow_redirects=True)
    with open(_METRICS_REFERENCE, "wb") as fileobj:
        fileobj.write(resp.content)


if not os.path.exists(_METRICS_REFERENCE):
    _download_metrics_reference()
