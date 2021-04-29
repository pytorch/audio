This is an example pipeline for speech recognition using a greedy or Viterbi CTC decoder, along with the Wav2Letter model trained on LibriSpeech, see [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf). Wav2Letter and LibriSpeech are available in torchaudio.

### Usage

More information about each command line parameters is available with the `--help` option. An example can be invoked as follows.
```bash
DATASET_ROOT = <Top>/<level>/<folder>
DATASET_FOLDER_IN_ARCHIVE =  'LibriSpeech'

python main.py \
    --reduce-lr-valid \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-folder-in-archive "${DATASET_FOLDER_IN_ARCHIVE}" \
    --dataset-train train-clean-100 train-clean-360 train-other-500 \
    --dataset-valid dev-clean \
    --batch-size 128 \
    --learning-rate .6 \
    --momentum .8 \
    --weight-decay .00001 \
    --clip-grad 0. \
    --gamma .99 \
    --hop-length 160 \
    --win-length 400 \
    --n-bins 13 \
    --normalize \
    --optimizer adadelta \
    --scheduler reduceonplateau \
    --epochs 40
```

With these default parameters, we get 13.3 %CER and 41.9 %WER on dev-clean after 40 epochs (character and word error rates, respectively) while training on train-clean. The tail of the output is the following.

```json
...
{"name": "train", "epoch": 40, "batch char error": 925, "batch char total": 22563, "batch char error rate": 0.040996321411159865, "epoch char error": 1135098.0, "epoch char total": 23857713.0, "epoch char error rate": 0.047577821059378154, "batch word error": 791, "batch word total": 4308, "batch word error rate": 0.18361188486536675, "epoch word error": 942906.0, "epoch word total": 4569507.0, "epoch word error rate": 0.20634742435015418, "lr": 0.06, "batch size": 128, "n_channel": 13, "n_time": 1685, "dataset length": 132096.0, "iteration": 1032.0, "loss": 0.07428030669689178, "cumulative loss": 90.47326805442572, "average loss": 0.08766789540157531, "iteration time": 1.9895553588867188, "epoch time": 2036.8874564170837}
{"name": "train", "epoch": 40, "batch char error": 1131, "batch char total": 24260, "batch char error rate": 0.0466199505358615, "epoch char error": 1136229.0, "epoch char total": 23881973.0, "epoch char error rate": 0.04757684802675223, "batch word error": 957, "batch word total": 4657, "batch word error rate": 0.2054971011380717, "epoch word error": 943863.0, "epoch word total": 4574164.0, "epoch word error rate": 0.20634655862798099, "lr": 0.06, "batch size": 128, "n_channel": 13, "n_time": 1641, "dataset length": 132224.0, "iteration": 1033.0, "loss": 0.08775319904088974, "cumulative loss": 90.5610212534666, "average loss": 0.08766797798012256, "iteration time": 2.108018159866333, "epoch time": 2038.99547457695}
{"name": "train", "epoch": 40, "batch char error": 1099, "batch char total": 23526, "batch char error rate": 0.0467142735696676, "epoch char error": 1137328.0, "epoch char total": 23905499.0, "epoch char error rate": 0.04757599914563591, "batch word error": 936, "batch word total": 4544, "batch word error rate": 0.20598591549295775, "epoch word error": 944799.0, "epoch word total": 4578708.0, "epoch word error rate": 0.20634620071863066, "lr": 0.06, "batch size": 128, "n_channel": 13, "n_time": 1682, "dataset length": 132352.0, "iteration": 1034.0, "loss": 0.0791337713599205, "cumulative loss": 90.64015502482653, "average loss": 0.08765972439538348, "iteration time": 2.0329701900482178, "epoch time": 2041.0284447669983}
{"name": "train", "epoch": 40, "batch char error": 1023, "batch char total": 22399, "batch char error rate": 0.045671681771507655, "epoch char error": 1138351.0, "epoch char total": 23927898.0, "epoch char error rate": 0.04757421650660664, "batch word error": 863, "batch word total": 4318, "batch word error rate": 0.1998610467809171, "epoch word error": 945662.0, "epoch word total": 4583026.0, "epoch word error rate": 0.20634009058643787, "lr": 0.06, "batch size": 128, "n_channel": 13, "n_time": 1644, "dataset length": 132480.0, "iteration": 1035.0, "loss": 0.07874362915754318, "cumulative loss": 90.71889865398407, "average loss": 0.08765110981061262, "iteration time": 1.9106628894805908, "epoch time": 2042.9391076564789}
{"name": "validation", "epoch": 40, "cumulative loss": 12.095281183719635, "dataset length": 2688.0, "iteration": 21.0, "batch char error": 1867, "batch char total": 14792, "batch char error rate": 0.12621687398593834, "epoch char error": 37119.0, "epoch char total": 280923.0, "epoch char error rate": 0.13213229247872194, "batch word error": 1155, "batch word total": 2841, "batch word error rate": 0.4065469904963041, "epoch word error": 22601.0, "epoch word total": 54008.0, "epoch word error rate": 0.418475040734706, "average loss": 0.575965770653316, "validation time": 24.185853481292725}
```
As can be seen in the output above, the information reported at each iteration and epoch (e.g. loss, character error rate, word error rate) is printed to standard output in the form of one json per line. One way to import the output in python with pandas is by saving the standard output to a file, and then using `pandas.read_json(filename, lines=True)`.

## Structure of pipeline

* `main.py` -- the entry point
* `ctc_decoders.py` -- the greedy CTC decoder
* `datasets.py` -- the function to split and process librispeech, a collate factory function
* `languagemodels.py` -- a class to encode and decode strings
* `metrics.py` -- the levenshtein edit distance
* `utils.py` -- functions to log metrics, save checkpoint, and count parameters
