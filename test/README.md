# Torchaudio Test Suite

## Structure of tests

The following is an overview of the tests and related modules for `torchaudio`.

### Purpose specific test suites

#### Numerical compatibility agains existing software
- [Librosa compatibility test](./test_librosa_compatibility.py)
    Test suite for numerical compatibility against librosa.
- [SoX compatibility test](./test_sox_compatibility.py)
    Test suite for numerical compatibility against SoX.
- [Kaldi compatibility test](./test_kaldi_compatibility.py)
    Test suite for numerical compatibility against Kaldi.

#### Result consistency with PyTorch framework
- [TorchScript consistency test](./test_torchscript_consistency.py)
    Test suite to check 1. if an API is TorchScript-able, and 2. the results from Python and Torchscript match.
- [Batch consistency test](./test_batch_consistency.py)
    Test suite to check if functionals/Transforms handle single sample input and batch input and return the same result.

### Module specific test suites

The following test modules are defined for corresponding `torchaudio` module/functions.

- [`torchaudio.datasets`](./test_datasets.py)
- [`torchaudio.functional`](./test_functional.py)
- [`torchaudio.transforms`](./test_transforms.py)
- [`torchaudio.compliance.kaldi`](./test_compliance_kaldi.py)
- [`torchaudio.kaldi_io`](./test_kaldi_io.py)
- [`torchaudio.sox_effects`](test/test_sox_effects.py)
- [`torchaudio.save`, `torchaudio.load`, `torchaudio.info`](test/test_io.py)

### Others (listing here for the sake of completeness)
- [test_dataloader.py](./test_dataloader.py)

### Support files
- [assets](./assets): Contain sample audio files.
- [assets/kaldi](./assets/kaldi): Contains Kaldi format matrix files used in [./test_compliance_kaldi.py](./test_compliance_kaldi.py).
- [compliance](./compliance): Scripts used to generate above Kaldi matrix files.


## Adding test

When you add a new feature(functional/transform), consider the following

1. When you add a new feature, please make it Torchscript-able and batch-consistent unless it degrades the performance. Please add the tests to see if the new feature meet these requirements.
1. If the feature should be numerical compatible against existing software (SoX, Librosa, Kaldi etc), add a corresponding test.
1. If the new feature is unique to `torchaudio` (not a PyTorch implementation of an existing Software functionality), consider adding correctness tests (wheather the expected output is produced for the set of input) under the corresponding test module (`test_functional.py`, `test_transforms.py`).
