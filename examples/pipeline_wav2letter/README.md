This is an example pipeline for speech recognition using a greedy or Viterbi CTC decoder, along with the Wav2Letter model trained on LibriSpeech, see [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf). Wav2Letter and LibriSpeech are available in torchaudio.

### Usage

More information about each command line parameters is available with the `--help` option. An example can be invoked as follows.
```
python main.py \
    --reduce-lr-valid \
    --dataset-train train-clean-100 train-clean-360 train-other-500 \
    --dataset-valid dev-clean \
    --batch-size 128 \
    --learning-rate .6 \
    --momentum .8 \
    --weight-decay .00001 \
    --clip-grad 0. \
    --gamma .99 \
    --hop-length 160 \
    --n-hidden-channels 2000 \
    --win-length 400 \
    --n-bins 13 \
    --normalize \
    --optimizer adadelta \
    --scheduler reduceonplateau \
    --epochs 30
```
With these default parameters, we get a character error rate of 13.8% on dev-clean after 30 epochs.

### Output

The information reported at each iteration and epoch (e.g. loss, character error rate, word error rate) is printed to standard output in the form of one json per line, e.g.
```
{"name": "train", "epoch": 0, "cer over target length": 1.0, "cumulative cer": 23317.0, "total chars": 23317.0, "cer": 0.0, "cumulative cer over target length": 0.0, "wer over target length": 1.0, "cumulative wer": 4446.0, "total words": 4446.0, "wer": 0.0, "cumulative wer over target length": 0.0, "lr": 0.6, "batch size": 128, "n_channel": 13, "n_time": 2453, "dataset length": 128.0, "iteration": 1.0, "loss": 8.712121963500977, "cumulative loss": 8.712121963500977, "average loss": 8.712121963500977, "iteration time": 41.46276903152466, "epoch time": 41.46276903152466}
{"name": "train", "epoch": 0, "cer over target length": 1.0, "cumulative cer": 46005.0, "total chars": 46005.0, "cer": 0.0, "cumulative cer over target length": 0.0, "wer over target length": 1.0, "cumulative wer": 8762.0, "total words": 8762.0, "wer": 0.0, "cumulative wer over target length": 0.0, "lr": 0.6, "batch size": 128, "n_channel": 13, "n_time": 1703, "dataset length": 256.0, "iteration": 2.0, "loss": 8.918599128723145, "cumulative loss": 17.63072109222412, "average loss": 8.81536054611206, "iteration time": 1.2905676364898682, "epoch time": 42.753336668014526}
{"name": "train", "epoch": 0, "cer over target length": 1.0, "cumulative cer": 70030.0, "total chars": 70030.0, "cer": 0.0, "cumulative cer over target length": 0.0, "wer over target length": 1.0, "cumulative wer": 13348.0, "total words": 13348.0, "wer": 0.0, "cumulative wer over target length": 0.0, "lr": 0.6, "batch size": 128, "n_channel": 13, "n_time": 1713, "dataset length": 384.0, "iteration": 3.0, "loss": 8.550191879272461, "cumulative loss": 26.180912971496582, "average loss": 8.726970990498861, "iteration time": 1.2109291553497314, "epoch time": 43.96426582336426}
```
One way to import the output in python with pandas is by saving the standard output to a file, and then using `pandas.read_json(filename, lines=True)`.

## Structure of pipeline

* `main.py` -- the entry point
* `ctc_decoders.py` -- the greedy CTC decoder
* `datasets.py` -- the function to split and process librispeech, a collate factory function
* `languagemodels.py` -- a class to encode and decode strings
* `metrics.py` -- the levenshtein edit distance
* `utils.py` -- functions to log metrics, save checkpoint, and count parameters
