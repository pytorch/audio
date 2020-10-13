This is an example pipeline for speech recognition using a greedy or Viterbi CTC decoder, along with the Wav2Letter model trained on LibriSpeech, see [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System](https://arxiv.org/pdf/1609.03193.pdf). Wav2Letter and LibriSpeech are available in torchaudio.

### Usage

More information about each command line parameters is available with the `--help` option. An example can be invoked as follows.
```
python main.py \
    --dataset-train train-clean-100 train-clean-360 train-other-500 \
    --dataset-valid dev-clean \
    --batch-size 128 \
    --hop-length 160 \
    --hidden-channels 2000 \
    --win-length 400 \
    --bins 13 \
    --normalize \
    --scheduler reduceonplateau \
    --reduce-lr-valid \
    --optimizer adadelta \
    --learning-rate .6 \
    --momentum .8 \
    --weight-decay .00001 \
    --clip-grad 0. \
    --max-epoch 40
```
With these default parameters, we get a character error rate of 13.8% on dev-clean after 30 epochs.

### Output

The information reported at each iteration and epoch (e.g. loss, character error rate, word error rate) is printed to standard output in the form of one json per line, e.g.
```
{"name": "train", "elapsed time": 26.613844830542803, "iteration": 1, "epoch": 0, "batch size": 128, "cumulative batch size": 128.0, "cumulative loss": 199767.09375, "epoch loss": 1560.680419921875, "batch loss": 1560.680419921875, "total chars": 22797.0, "cumulative char errors": 22670.0, "batch cer": 0.9944290915471334, "epoch cer": 0.9944290915471334, "total words": 4349.0, "cumulative word errors": 4320.0, "batch wer": 0.9933318004138882, "epoch wer": 0.9933318004138882, "lr": 0.6, "channel size": 13, "time size": 1669}
{"name": "train", "elapsed time": 14944777.921057917, "iteration": 2, "epoch": 0, "batch size": 128, "cumulative batch size": 256.0, "cumulative loss": 396356.84375, "epoch loss": 1548.2689208984375, "batch loss": 1535.857421875, "total chars": 46234.0, "cumulative char errors": 46107.0, "batch cer": 1.0, "epoch cer": 0.9972531037764416, "total words": 8805.0, "cumulative word errors": 8776.0, "batch wer": 1.0, "epoch wer": 0.9967064168086315, "lr": 0.6, "channel size": 13, "time size": 1677}
{"name": "train", "elapsed time": 21.74324156017974, "iteration": 1, "epoch": 0, "batch size": 128, "cumulative batch size": 128.0, "cumulative loss": 205090.8125, "epoch loss": 1602.27197265625, "batch loss": 1602.27197265625, "total chars": 23428.0, "cumulative char errors": 23300.0, "batch cer": 0.994536452108588, "epoch cer": 0.994536452108588, "total words": 4455.0, "cumulative word errors": 4414.0, "batch wer": 0.9907968574635241, "epoch wer": 0.9907968574635241, "lr": 0.6, "channel size": 13, "time size": 1697}
{"name": "train", "elapsed time": 1572825.2700412387, "iteration": 2, "epoch": 0, "batch size": 128, "cumulative batch size": 256.0, "cumulative loss": 402968.03125, "epoch loss": 1574.0938720703125, "batch loss": 1545.915771484375, "total chars": 46845.0, "cumulative char errors": 46717.0, "batch cer": 1.0, "epoch cer": 0.9972675845874693, "total words": 8887.0, "cumulative word errors": 8846.0, "batch wer": 1.0, "epoch wer": 0.9953865196354226, "lr": 0.6, "channel size": 13, "time size": 1652}
{"name": "train", "elapsed time": 36.98494444228709, "iteration": 3, "epoch": 0, "batch size": 128, "cumulative batch size": 384.0, "cumulative loss": 568859.890625, "epoch loss": 1481.4059651692708, "batch loss": 1347.6800537109375, "total chars": 69573.0, "cumulative char errors": 69446.0, "batch cer": 1.0, "epoch cer": 0.9981745792189499, "total words": 13265.0, "cumulative word errors": 13236.0, "batch wer": 1.0, "epoch wer": 0.9978137957029778, "lr": 0.6, "channel size": 13, "time size": 1690}
...
{"name": "validation", "elapsed time": 141.4699967801571, "iteration": 1, "epoch": 0, "batch size": 15, "cumulative batch size": 2703.0, "cumulative loss": 783672.287109375, "epoch loss": 289.9268542764983, "batch loss": 378.550390625, "total chars": 282969.0, "cumulative char errors": 198943.0, "batch cer": 0.6935483870967742, "epoch cer": 0.7030558117673668, "total words": 54402.0, "cumulative word errors": 63139.0, "batch wer": 1.2208121827411167, "epoch wer": 1.1606007132090732}
...
```
One way to import the output in python with pandas is by saving the standard output to a file, and then using `pandas.read_json(filename, lines=True)`.

## Structure of pipeline

* `main.py` -- command line entry point
* `engine.py` -- preprocessing, training, and validation, code
* `ctc_decoders.py` -- greedy CTC decoder
* `datasets.py` -- function to split and process librispeech, a collate factory function
* `languagemodels.py` -- class to encode and decode strings
* `metrics.py` -- levenshtein edit distance
* `utils.py` -- functions to log metrics, save checkpoint, and count parameters

## Distributed

The option `--distributed` enables distributed mode. For example with SLURM, one could use the follow file.
```
#SBATCH --job-name=torchaudiomodel
#SBATCH --open-mode=append
#SBATCH --nodes=2
#SBATCH --gres=gpu:8

export MASTER_ADDR=${SLURM_JOB_NODELIST:0:9}${SLURM_JOB_NODELIST:10:4}
export MASTER_PORT=29500

python main.py \
    --dataset-train train-clean-100 train-clean-360 train-other-500 \
    --dataset-valid dev-clean \
    --batch-size 128 \
    --hop-length 160 \
    --hidden-channels 2000 \
    --win-length 400 \
    --bins 13 \
    --normalize \
    --scheduler reduceonplateau \
    --reduce-lr-valid \
    --optimizer adadelta \
    --learning-rate .6 \
    --momentum .8 \
    --weight-decay .00001 \
    --clip-grad 0. \
    --max-epoch 40 \
    --distributed \
    --distributed-master-addr ${MASTER_ADDR} \
    --distributed-master-port ${MASTER_PORT}
```
