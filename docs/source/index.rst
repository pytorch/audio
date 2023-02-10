Torchaudio Documentation
========================

.. image:: _static/img/logo.png

Torchaudio is a library for audio and signal processing with PyTorch.
It provides I/O, signal and data processing functions, datasets,
model implementations and application components.

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
   build.ffmpeg

.. toctree::
   :maxdepth: 1
   :caption: API Tutorials
   :hidden:

   tutorials/audio_io_tutorial
   tutorials/streamreader_basic_tutorial
   tutorials/streamreader_advanced_tutorial
   tutorials/streamwriter_basic_tutorial
   tutorials/streamwriter_advanced
   hw_acceleration_tutorial

   tutorials/audio_resampling_tutorial
   tutorials/audio_data_augmentation_tutorial
   tutorials/audio_feature_extractions_tutorial
   tutorials/audio_feature_augmentation_tutorial

   tutorials/oscillator_tutorial
   tutorials/additive_synthesis_tutorial
   tutorials/filter_design_tutorial
   tutorials/subtractive_synthesis_tutorial

   tutorials/audio_datasets_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Pipeline Tutorials
   :hidden:

   tutorials/speech_recognition_pipeline_tutorial
   tutorials/asr_inference_with_ctc_decoder_tutorial
   tutorials/online_asr_tutorial
   tutorials/device_asr
   tutorials/forced_alignment_tutorial
   tutorials/tacotron2_pipeline_tutorial
   tutorials/mvdr_tutorial
   tutorials/hybrid_demucs_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Training Recipes
   :hidden:

   Conformer RNN-T ASR <https://github.com/pytorch/audio/tree/main/examples/asr/librispeech_conformer_rnnt>
   Emformer RNN-T ASR <https://github.com/pytorch/audio/tree/main/examples/asr/emformer_rnnt>
   Conv-TasNet Source Separation <https://github.com/pytorch/audio/tree/main/examples/source_separation>
   HuBERT Pre-training and Fine-tuning (ASR) <https://github.com/pytorch/audio/tree/main/examples/hubert>

.. toctree::
   :maxdepth: 1
   :caption: Python API Reference
   :hidden:

   torchaudio
   io
   backend
   functional
   transforms
   datasets
   models
   models.decoder
   pipelines
   sox_effects
   compliance.kaldi
   kaldi_io
   utils

.. toctree::
   :maxdepth: 1
   :caption: Python Prototype API Reference
   :hidden:

   prototype
   prototype.datasets
   prototype.functional
   prototype.models
   prototype.pipelines
   prototype.transforms

.. toctree::
   :maxdepth: 1
   :caption: C++ Prototype API Reference
   :hidden:

   libtorchaudio
   libtorchaudio.stream_reader
   libtorchaudio.stream_writer

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
   :header: Loading waveform Tensors from files and saving them
   :card_description: Learn how to query/load audio files and save waveform tensors to files, using <code>torchaudio.info</code>, <code>torchaudio.load</code> and <code>torchaudio.save</code> functions.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/audio_io_tutorial.png
   :link: tutorials/audio_io_tutorial.html
   :tags: I/O

.. customcarditem::
   :header: Streaming media decoding with StreamReader
   :card_description: Learn how to load audio/video to Tensors using <code>torchaudio.io.StreamReader</code> class.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/streamreader_basic_tutorial.png
   :link: tutorials/streamreader_basic_tutorial.html
   :tags: I/O,StreamReader

.. customcarditem::
   :header: Device input, synthetic audio/video, and filtering with StreamReader
   :card_description: Learn how to load media from hardware devices, generate synthetic audio/video, and apply filters to them with <code>torchaudio.io.StreamReader</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/streamreader_advanced_tutorial.gif
   :link: tutorials/streamreader_advanced_tutorial.html
   :tags: I/O,StreamReader

.. customcarditem::
   :header: Streaming media encoding with StreamWriter
   :card_description: Learn how to save audio/video with <code>torchaudio.io.StreamWriter</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/streamwriter_basic_tutorial.gif
   :link: tutorials/streamwriter_basic_tutorial.html
   :tags: I/O,StreamWriter
   
.. customcarditem::
   :header: Playing media with StreamWriter
   :card_description: Learn how to play audio/video with <code>torchaudio.io.StreamWriter</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/streamwriter_advanced.gif
   :link: tutorials/streamwriter_advanced.html
   :tags: I/O,StreamWriter

.. customcarditem::
   :header: Hardware accelerated video I/O with NVDEC/NVENC
   :card_description: Learn how to setup and use HW accelerated video I/O.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/hw_acceleration_tutorial.png
   :link: hw_acceleration_tutorial.html
   :tags: I/O,StreamReader,StreamWriter

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
   :header: Generating waveforms with oscillator
   :card_description:
   :image: _images/sphx_glr_oscillator_tutorial_003.png
   :link: tutorials/oscillator_tutorial.html
   :tags: DSP

.. customcarditem::
   :header: Additive Synthesis
   :card_description:
   :image: _images/sphx_glr_additive_synthesis_tutorial_001.png
   :link: tutorials/additive_synthesis_tutorial.html
   :tags: DSP

.. customcarditem::
   :header: Designing digital filters
   :card_description:
   :image: _images/sphx_glr_filter_design_tutorial_001.png
   :link: tutorials/filter_design_tutorial.html
   :tags: DSP

.. customcarditem::
   :header: Subtractive Synthesis
   :card_description:
   :image: _images/sphx_glr_subtractive_synthesis_tutorial_002.png
   :link: tutorials/subtractive_synthesis_tutorial.html
   :tags: DSP

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
   :header: Online ASR with Emformer RNN-T
   :card_description: Learn how to perform online ASR with Emformer RNN-T (<code>torchaudio.pipelines.RNNTBundle</code>) and <code>torchaudio.io.StreamReader</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/online_asr_tutorial.gif
   :link: tutorials/online_asr_tutorial.html
   :tags: Pipelines,ASR,RNNT,StreamReader
   
.. customcarditem::
   :header: Real-time microphone ASR with Emformer RNN-T
   :card_description: Learn how to transcribe speech fomr microphone with Emformer RNN-T (<code>torchaudio.pipelines.RNNTBundle</code>) and <code>torchaudio.io.StreamReader</code>.
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/thumbnails/device_asr.png
   :link: tutorials/device_asr.html
   :tags: Pipelines,ASR,RNNT,StreamReader

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

.. customcardend::


Citing torchaudio
-----------------

If you find torchaudio useful, please cite the following paper:

- Yang, Y.-Y., Hira, M., Ni, Z., Chourdia, A., Astafurov, A., Chen, C., Yeh, C.-F., Puhrsch, C.,
  Pollack, D., Genzel, D., Greenberg, D., Yang, E. Z., Lian, J., Mahadeokar, J., Hwang, J.,
  Chen, J., Goldsborough, P., Roy, P., Narenthiran, S., Watanabe, S., Chintala, S.,
  Quenneville-Bélair, V, & Shi, Y. (2021).
  TorchAudio: Building Blocks for Audio and Speech Processing. arXiv preprint arXiv:2110.15018.


In BibTeX format:

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
