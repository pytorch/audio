import json
import os

from transformers import Wav2Vec2Model

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _main():
    keys = [
        # pretrained
        "facebook/wav2vec2-base",
        "facebook/wav2vec2-large",
        "facebook/wav2vec2-large-lv60",
        "facebook/wav2vec2-base-10k-voxpopuli",
        "facebook/wav2vec2-large-xlsr-53",
        # finetuned
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h",
        "facebook/wav2vec2-large-960h-lv60",
        "facebook/wav2vec2-large-960h-lv60-self",
        "facebook/wav2vec2-large-xlsr-53-german",
    ]
    for key in keys:
        path = os.path.join(_THIS_DIR, f"{key}.json")
        print("Generating ", path)
        cfg = Wav2Vec2Model.from_pretrained(key).config
        cfg = json.loads(cfg.to_json_string())
        del cfg["_name_or_path"]

        with open(path, "w") as file_:
            file_.write(json.dumps(cfg, indent=4, sort_keys=True))
            file_.write("\n")


if __name__ == "__main__":
    _main()
