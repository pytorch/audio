"""Replaces every instance of 'torchaudio._backend' with 'torchaudio' in torchaudio.html.
Temporary hack while we maintain both the existing set of info/load/save functions and the
new ones backed by the backend dispatcher in torchaudio._backend.
"""

import sys

if __name__ == "__main__":
    build_dir = sys.argv[1]
    filepath = f"{build_dir}/html/torchaudio.html"

    with open(filepath, "r") as f:
        text = f.read()
        text = text.replace("torchaudio._backend", "torchaudio")

    with open(filepath, "w") as f:
        f.write(text)
