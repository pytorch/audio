# Torchaudio Test Suite

## How to run test

You can use `pytest` to run `torchaudio`'s test suites. See https://docs.pytest.org/ for the detail of how to use `pytest` command.

```
# List up all the tests
pytest test --collect-only
# Run all the test suites
pytest test
# Run tests on sox_effects module
pytest test/sox_effect
# use -k to apply filter
pytest test/sox_io_backend -k load  # only runs tests where their names contain load
# Some other useful options;
# Stop on the first failure -x
# Run failure fast --ff
# Only rerun the failure --lf
```

**Note**
We use PyTorch's test utilities instead of `pytest` frameworks when writing tests to avoid reinventing the wheel for Tensor comparison.

## Structure of tests

The following is an overview of the tests and related modules for `torchaudio`.

### Purpose specific test suites

#### Numerical compatibility agains existing software
- [Librosa compatibility test](./torchaudio_unittest/librosa_compatibility_test.py)
    Test suite for numerical compatibility against librosa.
- [SoX compatibility test](./torchaudio_unittest/sox_compatibility_test.py)
    Test suite for numerical compatibility against SoX.
- [Kaldi compatibility test](./torchaudio_unittest/kaldi_compatibility_test.py)
    Test suite for numerical compatibility against Kaldi.

#### Result consistency with PyTorch framework
- [TorchScript consistency test](./torchaudio_unittest/torchscript_consistency_impl.py)
    Test suite to check 1. if an API is TorchScript-able, and 2. the results from Python and Torchscript match.
- [Batch consistency test](./torchaudio_unittest/batch_consistency_test.py)
    Test suite to check if functionals/Transforms handle single sample input and batch input and return the same result.

### Module specific test suites

The following test modules are defined for corresponding `torchaudio` module/functions.

- [`torchaudio.datasets`](./torchaudio_unittest/datasets)
- [`torchaudio.functional`](./torchaudio_unittest/functional)
- [`torchaudio.transforms`](./torchaudio_unittest/transforms_test.py)
- [`torchaudio.compliance.kaldi`](./torchaudio_unittest/compliance_kaldi_test.py)
- [`torchaudio.kaldi_io`](./torchaudio_unittest/kaldi_io_test.py)
- [`torchaudio.sox_effects`](test/sox_effects)
- [`torchaudio.save`, `torchaudio.load`, `torchaudio.info`](./torchaudio_unittest/io_test.py)

### Test modules that do not fall into the above categories
- [test_dataloader.py](./torchaudio_unittest/dataloader_test.py)
    Simple test for loading data and applying preprocessing.

### Support files
- [assets](./torchaudio_unittest/assets): Contain sample audio files.
- [assets/kaldi](./torchaudio_unittest/assets/kaldi): Contains Kaldi format matrix files used in [./torchaudio_unittest/test_compliance_kaldi.py](./torchaudio_unittest/test_compliance_kaldi.py).
- [compliance](./torchaudio_unittest/compliance): Scripts used to generate above Kaldi matrix files.

### Waveforms for Testing Purposes

When testing transforms we often need waveforms of specific type (ex: pure tone, noise, or voice), with specific bitrate (ex. 8 or 16 kHz) and number of channels (ex. mono, stereo). Below are some tips on how to construct waveforms and guidance around existing audio files.

#### Load a Waveform from a File

```python
filepath = common_utils.get_asset_path('filename.wav')
waveform, sample_rate = common_utils.load_wav(filepath)
```

*Note: Should you choose to contribute an audio file, please leave a comment in the issue or pull request, mentioning content source and licensing information. WAV files are preferred. Other formats should be used only when there is no alternative. (i.e. dataset implementation comes with hardcoded non-wav extension).*

#### Pure Tone

Code:

```python
waveform = common_utils.get_sinusoid(
	frequency=300,
	sample_rate=16000,
	duration=1,  # seconds
	n_channels=1,
	dtype="float32",
	device="cpu",
)
```

#### Noise

Code:

```python
tensor = common_utils.get_whitenoise()
```

Files:

* `steam-train-whistle-daniel_simon.wav`

#### Voice

Files:

* `CommonVoice/cv-corpus-4-2019-12-10/tt/clips/common_voice_tt_00000000.wav`
* `VCTK-Corpus/wav48/p224/p224_002.wav`
* `vad-go-stereo-44100.wav`
* `vad-go-mono-32000.wav`

## Adding test

The following is the current practice of torchaudio test suite.

1. Unless the tests are related to I/O, use synthetic data. [`common_utils`](./torchaudio_unittest/common_utils) has some data generator functions.
1. When you add a new test case, use `common_utils.TorchaudioTestCase` as base class unless you are writing tests that are common to CPU / CUDA.
  - Set class memeber `dtype`, `device` and `backend` for the desired behavior.
  - If you do not set `backend` value in your test suite, then I/O functions will be unassigned and attempt to load/save file will fail.
  - For `backend` value, in addition to available backends, you can also provide the value "default" and backend will be picked automatically based on availability.
1. If you are writing tests that should pass on diffrent dtype/devices, write a common class inheriting `common_utils.TestBaseMixin`, then inherit `common_utils.PytorchTestCase` and define class attributes (`dtype` / `device` / `backend`) there. See [Torchscript consistency test implementation](./torchaudio_unittest/torchscript_consistency_impl.py) and test definitions for [CPU](./torchaudio_unittest/torchscript_consistency_cpu_test.py) and [CUDA](./torchaudio_unittest/torchscript_consistency_cuda_test.py) devices.
1. For numerically comparing Tensors, use `assertEqual` method from torchaudio_unittest.common_utils.PytorchTestCase` class. This method has a better support for a wide variety of Tensor types.

When you add a new feature(functional/transform), consider the following

1. When you add a new feature, please make it Torchscript-able and batch-consistent unless it degrades the performance. Please add the tests to see if the new feature meet these requirements.
1. If the feature should be numerical compatible against existing software (SoX, Librosa, Kaldi etc), add a corresponding test.
1. If the new feature is unique to `torchaudio` (not a PyTorch implementation of an existing Software functionality), consider adding correctness tests (wheather the expected output is produced for the set of input) under the corresponding test module (`test_functional.py`, `test_transforms.py`).
