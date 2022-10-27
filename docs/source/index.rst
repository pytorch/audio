Torchaudio Documentation
========================

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
   references

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
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

   tutorials/audio_datasets_tutorial

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
   :caption: API Reference
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
   :caption: Prototype API Reference
   :hidden:

   prototype
   prototype.functional
   prototype.models
   prototype.pipelines

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
   :card_description: Learn how to use query/load audio files and save waveform tensors to files.
   :image: _images/sphx_glr_audio_io_tutorial_001.png
   :link: tutorials/audio_io_tutorial.html
   :tags: I/O

.. customcarditem::
   :header: Streaming media decoding with StreamReader
   :card_description: Learn how to load audio/video to Tensors with StreamReader
   :image: _images/sphx_glr_streamreader_basic_tutorial_001.png
   :link: tutorials/streamreader_basic_tutorial.html
   :tags: I/O,StreamReader
   
.. customcarditem::
   :header: Advanced usage of StreamReader
   :card_description: Learn how to load media from hardware devices, generate synthetic audio/video, and filters.
   :image: /Users/moto/Development/torchaudio/docs/streamreader.gif
   :link: tutorials/streamreader_advanced_tutorial.html
   :tags: I/O,StreamReader

.. customcarditem::
   :header: Streaming media encoding with StreamWriter
   :card_description: Learn how to save audio/video with StreamWriter
   :image: /Users/moto/Development/torchaudio/docs/streamwriter.gif
   :link: tutorials/streamwriter_basic_tutorial.html
   :tags: I/O,StreamWriter
   
.. customcarditem::
   :header: Advanced usage of StreamWriter
   :card_description: Learn how to play audio/video with StreamWriter
   :image: /Users/moto/Development/torchaudio/docs/udp.gif
   :link: tutorials/streamwriter_advanced_tutorial.html
   :tags: I/O,StreamWriter

.. customcarditem::
   :header: Accelerated video I/O with NVDEC/NVENC
   :card_description: Learn how to setup HW acceleratoin for video processing
   :image: _images/hw_acceleration_tutorial_68_1.png
   :link: hw_acceleration_tutorial.html
   :tags: I/O,StreamReader,StreamWriter

.. customcarditem::
   :header: Audio resampling with bandlimited sinc interpolation  
   :card_description: Learn how to resample audio tensor
   :image: _images/sphx_glr_audio_resampling_tutorial_001.png
   :link: tutorials/audio_resampling_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio data augmentation
   :card_description: Learn how to use TorchAudio's functional/transforms modules to perform data augmentation
   :image: _images/sphx_glr_audio_data_augmentation_tutorial_007.png
   :link: tutorials/audio_data_augmentation_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio feature extraction
   :card_description: Learn how to use TorchAudio's functional/transforms modules to extract features from waveform
   :image: _images/sphx_glr_audio_feature_extractions_tutorial_011.png
   :link: tutorials/audio_feature_extractions_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio feature augmentation
   :card_description: Learn how to use TorchAudio's functional/transforms modules to peform feature augmentation
   :image: _images/sphx_glr_audio_feature_augmentation_tutorial_002.png
   :link: tutorials/audio_feature_augmentations_tutorial.html
   :tags: Preprocessing

.. customcarditem::
   :header: Audio dataset
   :card_description: Learn how to use TorchAudio's dataset module
   :image: _images/sphx_glr_audio_datasets_tutorial_001.png
   :link: tutorials/audio_datasets_tutorial.html
   :tags: Dataset

.. customcarditem::
   :header: AM inference with Wav2Vec2
   :card_description: Learn how to acoustice model inference with Wav2Vec2
   :image: _images/sphx_glr_speech_recognition_pipeline_tutorial_002.png
   :link: tutorials/speech_recognition_pipeline_tutorial.html
   :tags: ASR,wav2vec2

.. customcarditem::
   :header: LM inference with CTC Beam Seach Decoder
   :card_description: Learn how to lexicon, language model and beam search decoder in ASR
   :image: _images/sphx_glr_asr_inference_with_ctc_decoder_tutorial_001.png
   :link: tutorials/asr_inference_with_ctc_decoder_tutorial.html
   :tags: Pipelines,ASR,wav2vec2,CTC-Decoder,KenLM

.. customcarditem::
   :header: Online ASR with Emformer RNN-T
   :card_description: Learn how to perform online ASR with RNN-T and StreamReader
   :image: https://download.pytorch.org/torchaudio/tutorial-assets/emformer_rnnt_streamer_context.png
   :link: tutorials/online_asr_tutorial.html
   :tags: Pipelines,ASR,RNN-T,StreamReader
   
.. customcarditem::
   :header: Readl-time microphone ASR with Emformer RNN-T
   :card_description: Learn how to transcribe speech fomr micrphone with RNN-T and StreamReader
   :image: https://3.bp.blogspot.com/-aw5KRAWmnKE/WK7fGITeEqI/AAAAAAABCA4/j5G8qbugxmkdDWhmlAXy7ZYbTYhaaIv1ACLcB/s800/microphone_mark.png
   :link: tutorials/device_asr.html
   :tags: Pipelines,ASR,RNN-T,StreamReader

.. customcarditem::
   :header: Forced Alignment with Wav2Vec2
   :card_description: Learn how to align text to speech with Wav2Vec 2.0
   :image: _images/sphx_glr_forced_alignment_tutorial_005.png
   :link: tutorials/forced_alignment_tutorial.html
   :tags: Pipelines,Forced-Alignment,wav2vec2

.. customcarditem::
   :header: Text-to-Speech with Tacotron2
   :card_description: Learn how to generate speech from text with Tacotron2
   :image: _images/sphx_glr_tacotron2_pipeline_tutorial_003.png
   :link: tutorials/tacotron2_pipeline_tutorial.html
   :tags: Pipelines,Speech-to-Text
   
.. customcarditem::
   :header: Speech Enhancement with MVDR Beamforming
   :card_description: Learn how to improve speech quality with MVDR Beamforming
   :image: _images/sphx_glr_mvdr_tutorial_001.png
   :link: tutorials/mvdr_tutorial.html
   :tags: Pipelines,Speech-Enhancement

.. customcarditem::
   :header: Music Source Separation with Hybrid Demucs
   :card_description: Learn how to perform music separation with pre-trained Hybrid Demucs
   :image: _images/sphx_glr_hybrid_demucs_tutorial_001.png
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
