# Contextual Conformer RNN-T with TCPGen Example

This directory contains sample implementations of training and evaluation pipelines for the Conformer RNN-T model with tree-constrained pointer generator (TCPGen) for contextual biasing, as described in the paper: [Tree-Constrained Pointer Generator for End-to-End Contextual Speech Recognition](https://ieeexplore.ieee.org/abstract/document/9687915)

## Setup
### Install PyTorch and TorchAudio nightly or from source
Because Conformer RNN-T is currently a prototype feature, you will need to either use the TorchAudio nightly build or build TorchAudio from source. Note also that GPU support is required for training.

To install the nightly, follow the directions at <https://pytorch.org/>.

To build TorchAudio from source, refer to the [contributing guidelines](https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md).

### Install additional dependencies
```bash
pip install pytorch-lightning sentencepiece
```

## Usage

### Training

[`train.py`](./train.py) trains an Conformer RNN-T model with TCPGen on LibriSpeech using PyTorch Lightning. Note that the script expects users to have the following:
- Access to GPU nodes for training.
- Full LibriSpeech dataset.
- SentencePiece model to be used to encode targets; the model can be generated using [`train_spm.py`](./train_spm.py). **Note that suffix-based wordpieces are used in this example**. [`run_spm.sh`](./run_spm.sh) will generate 600 suffix-based wordpieces which is used in the paper.
- File (--global_stats_path) that contains training set feature statistics; this file can be generated using [`global_stats.py`](../emformer_rnnt/global_stats.py). The [`global_stats_100.json`](./global_stats_100.json) has been generated for train-clean-100.
- Biasing list: See [`rareword_f15.txt`](./blists/rareword_f15.txt) as an example for the biasing list used for training with clean-100 data. Words appeared less than 15 times were treated as rare words. For inference, [`all_rare_words.txt`](blists/all_rare_words.txt) which is the same list used in [https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias).

Sample local training script: [`train.sh`](./train.sh)

Other training options:
- `--droprate` is the drop rate of biasing words appeared in the reference text to avoid over-confidence
- `--maxsize` is the size of the biasing list used for training, which is the sum of biasing words in reference and distractors

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Conformer RNN-T model with TCPGen on LibriSpeech test-clean.

Sample decoding script: `eval.sh`

Other decoding options:

- `--biasinglist` should be [`all_rare_words.txt`](blists/all_rare_words.txt) for Librispeech experiments
- `--droprate` normally should be 0 because we assume the reference biasing words are included
- `--maxsize` is the size of the biasing list used for decoding, where 1000 was used in the paper.

### Scoring
Example scoring script using sclite is in [`score.sh`](./score.sh). Note that this will generate a file named `results.wrd.txt` which is in the format that will be used in the following script to calculate rare word error rate. Follow these steps to calculate rare word error rate:

```bash
cd error_analysis
python get_error_word_count.py <path_to_results.wrd.txt>
```

Note that the `word_freq.txt` file contains word frequencies for train-clean-100 only. For the full set it should be calculated again, which will only slightly affect OOV word error rate calculation in this case.
