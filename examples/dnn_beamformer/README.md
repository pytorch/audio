# Time-Frequency Mask based DNN MVDR Beamforming Example

This directory contains sample implementations of training and evaluation pipelines for an DNN Beamforming model.

The `DNNBeamformer` model composes the following componenst:

+ [`torchaudio.transforms.Spectrogram`](https://pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html#spectrogram) that applies Short-time Fourier Transform (STFT) to the waveform.
+ ConvTasNet without encoder/decoder that predicts T-F masks for speech and noise, respectively.
+ [`torchaudio.transforms.PSD`](https://pytorch.org/audio/stable/generated/torchaudio.transforms.PSD.html#psd) that computes covariance matrices for speech and noise.
+ [`torchaudio.transforms.SoudenMVDR`](https://pytorch.org/audio/stable/generated/torchaudio.transforms.SoudenMVDR.html#soudenmvdr) that estimates the compex-valued STFT for the enhanced speech.
+ [`torchaudio.transforms.InverseSpectrogram`](https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseSpectrogram.html#inversespectrogram) that applies inverse STFT (iSTFT) to generate the enhanced waveform.

## Usage

### Training

[`train.py`](./train.py) trains a [`DNNBeamformer`](./model.py) model using PyTorch Lightning. Note that the script expects users to have access to GPU nodes for training and provide paths to the [`L3DAS22`](https://www.kaggle.com/datasets/l3dasteam/l3das22) datasets.

### Evaluation

[`eval.py`](./eval.py) evaluates a trained [`DNNBeamformer`](./model.py) on the test subset of L3DAS22 dataset.

### L3DAS22

Sample SLURM command for training:
```
srun --cpus-per-task=12 --gpus-per-node=1 -N 1 --ntasks-per-node=1 python train.py --dataset-path ./datasets/L3DAS22 --checkpoint-path ./exp/checkpoints
```

Sample SLURM command for evaluation:
```
srun python eval.py --checkpoint-path ./exp/checkpoints/epoch=97-step=780472.ckpt --dataset-path ./datasets/L3DAS22 --use-cuda
```


Using the sample training command above, [`train.py`](./train.py) produces a model with 5.0M parameters (57.5MB).

The table below contains Ci-SDR, STOI, and PESQ results for the test subset of `L3DAS22` dataset.

|        Ci-SDR       |      STOI    |      PESQ    |
|:-------------------:|-------------:|-------------:|
|               19.00 |         0.82 |         2.46 |


If you find this training recipe useful, please cite as:

```bibtex
@article{yang2021torchaudio,
  title={TorchAudio: Building Blocks for Audio and Speech Processing},
  author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent Quenneville-Bélair and Yangyang Shi},
  journal={arXiv preprint arXiv:2110.15018},
  year={2021}
}

@inproceedings{lu2022towards,
  title={Towards low-distortion multi-channel speech enhancement: The ESPNet-SE submission to the L3DAS22 challenge},
  author={Lu, Yen-Ju and Cornell, Samuele and Chang, Xuankai and Zhang, Wangyou and Li, Chenda and Ni, Zhaoheng and Wang, Zhong-Qiu and Watanabe, Shinji},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={9201--9205},
  year={2022},
  organization={IEEE}
}
```
