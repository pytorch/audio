This is an example vocoder pipeline using the WaveRNN model trained with LJSpeech. WaveRNN model is based on the implementation from [this repository](https://github.com/fatchord/WaveRNN). The original implementation was
introduced in "Efficient Neural Audio Synthesis". WaveRNN and LJSpeech are available in torchaudio.

### Usage

An example can be invoked as follows.
```
python main.py \
    --batch-size 256 \
    --learning-rate 1e-4 \
    --n-freq 80 \
    --loss 'crossentropy' \
    --n-bits 8 \
```

For inference, an example can be invoked as follows.
Please refer to the [documentation](https://pytorch.org/audio/master/models.html#id10) for
available checkpoints.
```
python inference.py \
    --checkpoint-name wavernn_10k_epochs_8bits_ljspeech \
    --output-wav-path ./output.wav
```

This example would generate a file named `output.wav` in the current working directory.

### Output

The information reported at each iteration and epoch (e.g. loss) is printed to standard output in the form of one json per line. Here is an example python function to parse the output if redirected to a file.
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
