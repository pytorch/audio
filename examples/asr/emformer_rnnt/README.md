# Emformer RNN-T ASR Example

This directory contains sample implementations of training and evaluation pipelines for an Emformer RNN-T streaming ASR model.

## Usage

### Training

[`train.py`](./train.py) trains an Emformer RNN-T model using PyTorch Lightning. Note that the script expects users to have access to GPU nodes for training and provide paths to datasets and the SentencePiece model to be used to encode targets. The script also expects a file (--global_stats_path) that contains training set feature statistics; this file can be generated via [`global_stats.py`](./global_stats.py).

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Emformer RNN-T model on a given dataset.

### Pipeline Demo

[`pipeline_demo.py`](./pipeline_demo.py) demonstrates how to use the `EMFORMER_RNNT_BASE_LIBRISPEECH`
or `EMFORMER_RNNT_BASE_TEDLIUM3` bundle that wraps a pre-trained Emformer RNN-T produced by the corresponding recipe below to perform streaming and full-context ASR on several audio samples.

## Model Types

Currently, we have training recipes for the LibriSpeech and TED-LIUM Release 3 datasets.

### LibriSpeech

Sample SLURM command for training:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 4 --ntasks-per-node=8 python train.py --model_type librispeech --exp_dir ./experiments --dataset_path ./datasets/librispeech --global_stats_path ./global_stats.json --sp_model_path ./spm_bpe_4096.model
```

Sample SLURM command for evaluation:
```
srun python eval.py --model_type librispeech --checkpoint_path ./experiments/checkpoints/epoch=119-step=208079.ckpt --dataset_path ./datasets/librispeech  --sp_model_path ./spm_bpe_4096.model --use_cuda
```

The script used for training the SentencePiece model that's referenced by the training command above can be found at [`librispeech/train_spm.py`](./librispeech/train_spm.py); a pretrained SentencePiece model can be downloaded [here](https://download.pytorch.org/torchaudio/pipeline-assets/spm_bpe_4096_librispeech.model).

Using the sample training command above, [`train.py`](./train.py) produces a model with 76.7M parameters (307MB) that achieves an WER of 0.0456 when evaluated on test-clean with [`eval.py`](./eval.py).

The table below contains WER results for various splits.

|                     |          WER |
|:-------------------:|-------------:|
| test-clean          |       0.0456 |
| test-other          |       0.1066 |
| dev-clean           |       0.0415 |
| dev-other           |       0.1110 |

### TED-LIUM Release 3

Whereas the LibriSpeech model is configured with a vocabulary size of 4096, the TED-LIUM Release 3 model is configured with a vocabulary size of 500. Consequently, the TED-LIUM Release 3 model's last linear layer in the joiner has an output dimension of 501 (500 + 1 to account for the blank symbol); the rest of the model is identical to the LibriSpeech model.

Sample SLURM command for training:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 1 --ntasks-per-node=8 python train.py --model_type tedlium3 --exp_dir ./experiments --dataset_path ./datasets/tedlium --global_stats_path ./global_stats.json --sp_model_path ./spm_bpe_500.model --gradient_clip_val 5.0
```

Sample SLURM command for evaluation:
```
srun python eval.py --model_type tedlium3 --checkpoint_path ./experiments/checkpoints/epoch=119-step=254999.ckpt  --tedlium_path ./datasets/tedlium --sp_model_path ./spm-bpe-500.model --use_cuda
```

The table below contains WER results for dev and test subsets of TED-LIUM release 3.

|             |          WER |
|:-----------:|-------------:|
| dev         |       0.108  |
| test        |       0.098  |

[`tedlium3/eval_pipeline.py`](./tedlium3/eval_pipeline.py) evaluates the pre-trained `EMFORMER_RNNT_BASE_TEDLIUM3` bundle on the dev and test sets of TED-LIUM release 3. Running the script should produce WER results that are identical to those in the above table.
