This is an example pipeline for speech recognition using a greedy or Viterbi CTC decoder, along with the Wav2Letter model trained on LibriSpeech, see [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf). Wav2Letter and LibriSpeech are available in torchaudio.

### Usage

More information about each command line parameters is available with the `--help` option. An example can be invoked as follows.
```
python main.py \
    --batch-size 128 \
    --learning-rate .6 \
    --gamma .99 \
    --n-bins 13 \
    --momentum .8 \
    --clip-grad 0. \
    --optimizer "adadelta" \
    --scheduler "reduceonplateau"
```

### Output

The information reported at each iteration and epoch (e.g. loss, character error rate, word error rate) is printed to standard output in the form of one json per line. Further information is reported to standard error. Here is an example python function to parse the standard output when saved to a file.
```python
def read_json(filename):
	"""
	Convert the standard output saved to filename into a pandas dataframe for analysis.
	"""

	import pandas
	import json

    with open(filename, "r") as f:
        data = f.read()

    # pandas doesn't read single quotes for json
    data = data.replace("'", '"')

    data = [json.loads(l) for l in data.splitlines()]
    return pandas.DataFrame(data)
```

## Structure of pipeline

* `main.py` -- the entry point
* `ctc_decoders.py` -- the greedy CTC decoder
* `datasets.py` -- the function to split and process librispeech, a collate factory function
* `languagemodels.py` -- a class to encode and decode strings
* `metrics.py` -- the levenshtein edit distance
* `utils.py` -- functions to log metrics, save checkpoint, and count parameters
