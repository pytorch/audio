torchaudio: an audio library for PyTorch
========================================

[![Documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchaudio%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/audio/main/)
[![Anaconda Badge](https://anaconda.org/pytorch/torchaudio/badges/downloads.svg)](https://anaconda.org/pytorch/torchaudio)
[![Anaconda-Server Badge](https://anaconda.org/pytorch/torchaudio/badges/platforms.svg)](https://anaconda.org/pytorch/torchaudio)

![TorchAudio Logo](docs/source/_static/img/logo.png)

> [!NOTE]
> **We have transitioned TorchAudio into a
>  maintenance phase. This process removed some user-facing
>  features. These features were deprecated from TorchAudio 2.8 and removed in 2.9.
>  Our main goals were to reduce redundancies with the rest of the
>  PyTorch ecosystem, make it easier to maintain, and create a version of
>  TorchAudio that is more tightly scoped to its strengths: processing audio
>  data for ML. Please see
>  [our community message](https://github.com/pytorch/audio/issues/3902)
>  for more details.**

The aim of torchaudio is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the audio domain. By supporting PyTorch, torchaudio follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of PyTorch can be seen in torchaudio through
having all the computations be through PyTorch operations which makes it easy
to use and feel like a natural extension.

- [Dataloaders for common audio datasets](http://pytorch.org/audio/main/datasets.html)
- Audio and speech processing functions
  - [forced_align](https://pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html)
- Common audio transforms
  - [Spectrogram, AmplitudeToDB, MelScale, MelSpectrogram, MFCC, MuLawEncoding, MuLawDecoding, Resample](http://pytorch.org/audio/main/transforms.html)
- Compliance interfaces: Run code using PyTorch that align with other libraries
  - [Kaldi: spectrogram, fbank, mfcc](https://pytorch.org/audio/main/compliance.kaldi.html)

Installation
------------

Please refer to https://pytorch.org/audio/main/installation.html for installation and build process of TorchAudio.


API Reference
-------------

API Reference is located here: http://pytorch.org/audio/main/

Contributing Guidelines
-----------------------

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

Citation
--------

If you find this package useful, please cite as:

```bibtex
@article{yang2021torchaudio,
  title={TorchAudio: Building Blocks for Audio and Speech Processing},
  author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent Quenneville-Bélair and Yangyang Shi},
  journal={arXiv preprint arXiv:2110.15018},
  year={2021}
}
```

```bibtex
@misc{hwang2023torchaudio,
      title={TorchAudio 2.1: Advancing speech recognition, self-supervised learning, and audio processing components for PyTorch},
      author={Jeff Hwang and Moto Hira and Caroline Chen and Xiaohui Zhang and Zhaoheng Ni and Guangzhi Sun and Pingchuan Ma and Ruizhe Huang and Vineel Pratap and Yuekai Zhang and Anurag Kumar and Chin-Yun Yu and Chuang Zhu and Chunxi Liu and Jacob Kahn and Mirco Ravanelli and Peng Sun and Shinji Watanabe and Yangyang Shi and Yumeng Tao and Robin Scheibler and Samuele Cornell and Sean Kim and Stavros Petridis},
      year={2023},
      eprint={2310.17864},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

Disclaimer on Datasets
----------------------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!

Pre-trained Model License
-------------------------

The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission to use the models for your use case.

For instance, SquimSubjective model is released under the Creative Commons Attribution Non Commercial 4.0 International (CC-BY-NC 4.0) license. See [the link](https://zenodo.org/record/4660670#.ZBtWPOxuerN) for additional details.

Other pre-trained models that have different license are noted in documentation. Please checkout the [documentation page](https://pytorch.org/audio/main/).

Audio augmentation using sox_effect
-------------------------
Step 1: Install torchaudio            
pip install torchaudio   

Step 2: Import Necessary Libraries
import torchaudio
import torchaudio.sox_effects as sox_effects
      
Step 3: Load an Audio File  
waveform, sample_rate = torchaudio.load("path_to_audio_file")    
 
Step 4: Define the SoX Effects and apply the effects to input audio. In this example, I apply time_stretch, loudness, and high-pass filter adjustment.
   
effects = [
  ["tempo", "1.25"], # Increase the playback speed (tempo) by 25%  
  ["gain", "10"], # Amplify the audio by 10 dB  
  ["highpass", "1000"], # Apply a high-pass filter with a cutoff frequency of 1000 Hz  
]

augmented_waveform, augmented_sample_rate = sox_effects.apply_effects_tensor(
    waveform,     
    sample_rate,  
    effects  
)

Step 5: Save the Augmented Audio

torchaudio.save("augmented_audio.wav", augmented_waveform, augmented_sample_rate)
