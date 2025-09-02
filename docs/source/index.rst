Torchaudio Documentation
========================

.. image:: _static/img/logo.png

Torchaudio is a library for audio and signal processing with PyTorch.
It provides signal and data processing functions, datasets,
model implementations and application components.

.. note::
    Starting with version 2.8, we have transitioned into a maintenance phase. As a result:

    - Some APIs were deprecated in 2.8 and removed as of 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      have been consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.


..
   Generate Table Of Contents (left navigation bar)
   NOTE: If you are adding tutorials, add entries to toctree and customcarditem below

.. toctree::
   :maxdepth: 1
   :caption: Torchaudio Documentation
   :hidden:

   Index <self>
   supported_features
   feature_classifications
   logo
   references

.. toctree::
   :maxdepth: 2
   :caption: Installation
   :hidden:

   installation
   build
   build.linux
   build.windows
   build.jetson

.. toctree::
   :maxdepth: 1
   :caption: Training Recipes
   :hidden:

   Conformer RNN-T ASR <https://github.com/pytorch/audio/tree/main/examples/asr/librispeech_conformer_rnnt>
   Emformer RNN-T ASR <https://github.com/pytorch/audio/tree/main/examples/asr/emformer_rnnt>
   Conv-TasNet Source Separation <https://github.com/pytorch/audio/tree/main/examples/source_separation>
   HuBERT Pre-training and Fine-tuning (ASR) <https://github.com/pytorch/audio/tree/main/examples/hubert>
   Real-time AV-ASR <https://github.com/pytorch/audio/tree/main/examples/avsr>

.. toctree::
   :maxdepth: 1
   :caption: Python API Reference
   :hidden:

   torchaudio
   functional
   transforms
   datasets
   models
   models.decoder
   compliance.kaldi
   pipelines

.. toctree::
   :maxdepth: 1
   :caption: PyTorch Libraries
   :hidden:

   PyTorch <https://pytorch.org/docs>
   torchaudio <https://pytorch.org/audio>
   torchtext <https://pytorch.org/text>
   torchvision <https://pytorch.org/vision>
   TorchElastic <https://pytorch.org/elastic/>
   TorchServe <https://pytorch.org/serve>
   PyTorch on XLA Devices <http://pytorch.org/xla/>

Tutorials
---------

.. customcardstart::

.. customcarditem::
   :header: AM inference with CUDA CTC Beam Seach Decoder
   :card_description: Learn how to perform ASR beam search decoding with GPU, using <code>torchaudio.models.decoder.cuda_ctc_decoder</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/asr_inference_with_ctc_decoder_tutorial.png
   :link: tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html
   :tags: Pipelines,ASR,CTC-Decoder,CUDA-CTC-Decoder

.. customcarditem::
   :header: CTC Forced Alignment API
   :card_description: Learn how to use TorchAudio's CTC forced alignment API (<code>torchaudio.functional.forced_align</code>).
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/ctc_forced_alignment_api_tutorial.png
   :link: tutorials/ctc_forced_alignment_api_tutorial.html
   :tags: CTC,Forced-Alignment

.. customcarditem::
   :header: Forced alignment for multilingual data
   :card_description: Learn how to use align multiligual data using TorchAudio's CTC forced alignment API (<code>torchaudio.functional.forced_align</code>) and a multiligual Wav2Vec2 model.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/forced_alignment_for_multilingual_data_tutorial.png
   :link: tutorials/forced_alignment_for_multilingual_data_tutorial.html
   :tags: Forced-Alignment

.. customcarditem::
   :header: Audio resampling with bandlimited sinc interpolation
   :card_description: Learn how to resample audio tensor with <code>torchaudio.functional.resample</code> and <code>torchaudio.transforms.Resample</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/audio_resampling_tutorial.png
   :link: tutorials/audio_resampling_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio data augmentation
   :card_description: Learn how to use <code>torchaudio.functional</code> and <code>torchaudio.transforms</code> modules to perform data augmentation.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/audio_data_augmentation_tutorial.png
   :link: tutorials/audio_data_augmentation_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio feature extraction
   :card_description: Learn how to use <code>torchaudio.functional</code> and <code>torchaudio.transforms</code> modules to extract features from waveform.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/audio_feature_extractions_tutorial.png
   :link: tutorials/audio_feature_extractions_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio feature augmentation
   :card_description: Learn how to use <code>torchaudio.functional</code> and <code>torchaudio.transforms</code> modules to perform feature augmentation.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/audio_feature_augmentation_tutorial.png
   :link: tutorials/audio_feature_augmentation_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio dataset
   :card_description: Learn how to use <code>torchaudio.datasets</code> module.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/audio_datasets_tutorial.png
   :link: tutorials/audio_datasets_tutorial.html
   :tags: Dataset

.. customcarditem::
   :header: AM inference with Wav2Vec2
   :card_description: Learn how to perform acoustic model inference with Wav2Vec2 (<code>torchaudio.pipelines.Wav2Vec2ASRBundle</code>).
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/speech_recognition_pipeline_tutorial.png
   :link: tutorials/speech_recognition_pipeline_tutorial.html
   :tags: ASR,wav2vec2

.. customcarditem::
   :header: LM inference with CTC Beam Seach Decoder
   :card_description: Learn how to perform ASR beam search decoding with lexicon and language model, using <code>torchaudio.models.decoder.ctc_decoder</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/asr_inference_with_ctc_decoder_tutorial.png
   :link: tutorials/asr_inference_with_ctc_decoder_tutorial.html
   :tags: Pipelines,ASR,wav2vec2,CTC-Decoder

.. customcarditem::
   :header: Forced Alignment with Wav2Vec2
   :card_description: Learn how to align text to speech with Wav2Vec 2 (<code>torchaudio.pipelines.Wav2Vec2ASRBundle</code>).
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/forced_alignment_tutorial.png
   :link: tutorials/forced_alignment_tutorial.html
   :tags: Pipelines,Forced-Alignment,wav2vec2

.. customcarditem::
   :header: Text-to-Speech with Tacotron2
   :card_description: Learn how to generate speech from text with Tacotron2 (<code>torchaudio.pipelines.Tacotron2TTSBundle</code>).
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/tacotron2_pipeline_tutorial.png
   :link: tutorials/tacotron2_pipeline_tutorial.html
   :tags: Pipelines,TTS-(Text-to-Speech)

.. customcarditem::
   :header: Speech Enhancement with MVDR Beamforming
   :card_description: Learn how to improve speech quality with MVDR Beamforming.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/mvdr_tutorial.png
   :link: tutorials/mvdr_tutorial.html
   :tags: Pipelines,Speech-Enhancement

.. customcarditem::
   :header: Music Source Separation with Hybrid Demucs
   :card_description: Learn how to perform music source separation with pre-trained Hybrid Demucs (<code>torchaudio.pipelines.SourceSeparationBundle</code>).
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/hybrid_demucs_tutorial.png
   :link: tutorials/hybrid_demucs_tutorial.html
   :tags: Pipelines,Source-Separation

.. customcarditem::
   :header: Torchaudio-Squim: Non-intrusive Speech Assessment in TorchAudio
   :card_description: Learn how to estimate subjective and objective metrics with pre-trained TorchAudio-SQUIM models (<code>torchaudio.pipelines.SQUIMObjective</code>).
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/squim_tutorial.png
   :link: tutorials/squim_tutorial.html
   :tags: Pipelines,Speech Assessment,Speech Enhancement
.. customcardend::


Citing torchaudio
-----------------

If you find torchaudio useful, please cite the following paper:

-  Hwang, J., Hira, M., Chen, C., Zhang, X., Ni, Z., Sun, G., Ma, P., Huang, R., Pratap, V.,
   Zhang, Y., Kumar, A., Yu, C.-Y., Zhu, C., Liu, C., Kahn, J., Ravanelli, M., Sun, P.,
   Watanabe, S., Shi, Y., Tao, T., Scheibler, R., Cornell, S., Kim, S., & Petridis, S. (2023).
   TorchAudio 2.1: Advancing speech recognition, self-supervised learning, and audio processing components for PyTorch. arXiv preprint arXiv:2310.17864

- Yang, Y.-Y., Hira, M., Ni, Z., Chourdia, A., Astafurov, A., Chen, C., Yeh, C.-F., Puhrsch, C.,
  Pollack, D., Genzel, D., Greenberg, D., Yang, E. Z., Lian, J., Mahadeokar, J., Hwang, J.,
  Chen, J., Goldsborough, P., Roy, P., Narenthiran, S., Watanabe, S., Chintala, S.,
  Quenneville-Bélair, V, & Shi, Y. (2021).
  TorchAudio: Building Blocks for Audio and Speech Processing. arXiv preprint arXiv:2110.15018.

In BibTeX format:

.. code-block:: bibtex

   @misc{hwang2023torchaudio,
      title={TorchAudio 2.1: Advancing speech recognition, self-supervised learning, and audio processing components for PyTorch},
      author={Jeff Hwang and Moto Hira and Caroline Chen and Xiaohui Zhang and Zhaoheng Ni and Guangzhi Sun and Pingchuan Ma and Ruizhe Huang and Vineel Pratap and Yuekai Zhang and Anurag Kumar and Chin-Yun Yu and Chuang Zhu and Chunxi Liu and Jacob Kahn and Mirco Ravanelli and Peng Sun and Shinji Watanabe and Yangyang Shi and Yumeng Tao and Robin Scheibler and Samuele Cornell and Sean Kim and Stavros Petridis},
      year={2023},
      eprint={2310.17864},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
   }

.. code-block:: bibtex

    @article{yang2021torchaudio,
      title={TorchAudio: Building Blocks for Audio and Speech Processing},
      author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and
              Anjali Chourdia and Artyom Astafurov and Caroline Chen and
              Ching-Feng Yeh and Christian Puhrsch and David Pollack and
              Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and
              Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and
              Peter Goldsborough and Prabhat Roy and Sean Narenthiran and
              Shinji Watanabe and Soumith Chintala and
              Vincent Quenneville-Bélair and Yangyang Shi},
      journal={arXiv preprint arXiv:2110.15018},
      year={2021}
    }
