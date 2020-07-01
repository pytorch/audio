This is an example pipeline for speech recognition using a greedy or Viterbi CTC decoder, along with the Wav2Letter model trained on LibriSpeech. Wav2Letter and LibriSpeech are available in torchaudio.

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
    --scheduler "exponential"
```

### Output

The information reported at each iteration and epoch (e.g. loss, character error rate, word error rate) is printed to standard output in the form of one json per line. Further information is reported to standard error. Here is an example python function to parse the standard output.
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
