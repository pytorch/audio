This is an example pipeline for WaveRNN vocoder using the WaveRNN model trained with LJSpeech. WaveRNN model is based on the implementation from this repository https://github.com/fatchord/WaveRNN. The original implementation was
introduced in "Efficient Neural Audio Synthesis". WaveRNN and LJSpeech are available in torchaudio.

### Output

The information reported at each iteration and epoch (e.g. loss) is printed to standard output in the form of one json per line. Here is an example python function to parse the standard output.
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

### Usage

More information about each command line parameters is available with the `--help` option. An example can be invoked as follows.
```
python main.py \
    --batch-size 128 \
    --learning-rate 1e-4 \
    --n-freq 80 \
    --mode 'waveform' \
    --n-bits 8 \
```
