# Contributing to Torchaudio
We want to make contributing to this project as easy and transparent as possible.

## TL;DR

Please let us know if you encounter a bug by filing an [issue](https://github.com/pytorch/audio/issues).

We appreciate all contributions. If you are planning to contribute back
bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the
core, please first open an issue and discuss the feature with us. Sending a PR
without discussion might end up resulting in a rejected PR, because we might be
taking the core in a different direction than you might be aware of.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the
safe disclosure of security bugs. In those cases, please go through the
process outlined on that page and do not file a public issue.

Fixing bugs and implementing new features are not the only way you can
contribute. It also helps the project when you report problems you're facing,
and when you give a :+1: on issues that others reported and that are relevant
to you.

You can also help by improving the documentation. This is no less important
than improving the library itself! If you find a typo in the documentation,
do not hesitate to submit a pull request.

If you're not sure what you want to work on, you can pick an issue from the
[list of open issues labelled as "help
wanted"](https://github.com/pytorch/audio/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).
Comment on the issue that you want to work on it and send a PR with your fix
(see below).

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Building torchaudio from source

Please refer to https://pytorch.org/audio/main/build.html

## Running Test

If you built sox, set the `PATH` variable so that the tests properly use the newly built `sox` binary:

```bash
export PATH="<path_to_torchaudio>/third_party/install/bin:${PATH}"
```

The following dependencies are also needed for testing:

```bash
pip install typing pytest scipy numpy parameterized
```

Optional packages to install if you want to run related tests:

- `librosa`
- `requests`
- `soundfile`
- `transformers`
- `fairseq` (it has to be newer than `0.10.2`, so you will need to install from
  source. Commit `e6eddd80` is known to work.)
- `unidecode` (dependency for testing text preprocessing functions for examples/pipeline_tacotron2)
- `inflect` (dependency for testing text preprocessing functions for examples/pipeline_tacotron2)

## Development Process

If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `main`: `$ git checkout main && git checkout -b my_cool_feature`
2. If you have modified the code (new feature or bug-fix), [please add tests](test/torchaudio_unittest/).
3. If you have changed APIs, [update the documentation](#Documentation).

For more details about pull requests,
please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

If you would like to contribute a new model, please see [here](#New-model).

If you would like to contribute a new dataset, please see [here](#New-dataset).

## Testing

Please refer to our [testing guidelines](test/torchaudio_unittest/) for more
details.

## Documentation

Torchaudio uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 120 characters.

To build the docs, first install the requirements:

```bash
cd docs
pip install -r requirements.txt
```

Then:

```bash
cd docs
make html
```

The built docs should now be available in `docs/build/html`.
If docstrings are mal-formed, warnings will be shown.
In CI doc build job, `SPHINXOPTS=-W` option is enabled and warnings are treated as error.
Please fix all the warnings when submitting a PR.
(You can use `SPHINXOPTS=-W` in local env, but by default,
tutorials are not built and it will be treated as error.
To use the option, please set  `BUILD_GALLERY` as well.
e.g. `BUILD_GALLERY=1 make 'SPHINXOPTS=-W' html`.)

By default, the documentation only builds API reference.
If you are working to add a new example/tutorial with sphinx-gallery then
install the additional packages and set `BUILD_GALLERY` environment variable.

```bash
pip install -r requirements-tutorials.txt
BUILD_GALLERY=1 make html
```

This will build all the tutorials with ending `_tutorial.py`.
This can be time consuming. You can further filter which tutorial to build by using
`GALLERY_PATTERN` environment variable.

```
BUILD_GALLERY=1 GALLERY_PATTERN=forced_alignment_tutorial.py make html
```

Omitting `BUILD_GALLERY` while providing `GALLERY_PATTERN` assumes `BUILD_GALLERY=1`.

```
GALLERY_PATTERN=forced_alignment_tutorial.py make html
```

## Adding a new tutorial

We use Sphinx-Gallery to generate tutorials. Please refer to the [documentation](https://sphinx-gallery.github.io/stable/syntax.html) for how to format the tutorial.

You can draft in Google Colab and export it as IPython notebook and use [this script](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe) to convert it to Python file, but this process is known to incur some rendering issue. So please make sure to the resulting tutorial renders correctly.

Some tips;

- Use the suffix `_tutorial.py` to be recognized by the doc build process.
- When displaying audio with `IPython.display.Audio`, put one audio object per cell and put it at the end so that the resulting audio is embedded. (https://github.com/pytorch/audio/pull/1985)
- Similarly, when adding plots, add one plot per one code cell (use `subplots` to plot multiple), so that the resulting image is properly picked up.
- Avoid using `=` for section header, use `-` or `~`. Otherwise the resulting doc will have an issue like https://github.com/pytorch/audio/pull/1989.

## Conventions

As a good software development practice, we try to stick to existing variable
names and shape (for tensors), and maintain consistent docstring standards.
The following are some of the conventions that we follow.

- Tensor
  - We use an ellipsis "..." as a placeholder for the rest of the dimensions of a
    tensor, e.g. optional batching and channel dimensions. If batching, the
    "batch" dimension should come in the first dimension.
  - Tensors are assumed to have "channel" dimension coming before the "time"
    dimension. The bins in frequency domain (freq and mel) are assumed to come
    before the "time" dimension but after the "channel" dimension. These
    ordering makes the tensors consistent with PyTorch's dimensions.
  - For size names, the prefix `n_` is used (e.g. "a tensor of size (`n_freq`,
    `n_mels`)") whereas dimension names do not have this prefix (e.g. "a tensor of
    dimension (channel, time)")
- Docstring
  - Tensor dimensions are enclosed with single backticks.
    ``waveform (Tensor): Tensor of audio of dimension `(..., time)` ``
  - Parameter type for variable of type `T` with a default value: `(T, optional)`
  - Parameter type for variable of type `Optional[T]`: `(T or None)`
  - Return type for a tuple or list of known elements:
    `(element1, element2)` or `[element1, element2]`
  - Return type for a tuple or list with an arbitrary number of elements
    of type T: `Tuple[T]` or `List[T]`

Here are some of the examples of commonly used variables with thier names,
meanings, and shapes (or units):

* `waveform`: a tensor of audio samples with dimensions `(..., channel, time)`
* `sample_rate`: the rate of audio dimensions `(samples per second)`
* `specgram`: a tensor of spectrogram with dimensions `(..., channel, freq, time)`
* `mel_specgram`: a mel spectrogram with dimensions `(..., channel, mel, time)`
* `hop_length`: the number of samples between the starts of consecutive frames
* `n_fft`: the number of Fourier bins
* `n_mels`, `n_mfcc`: the number of mel and MFCC bins
* `n_freq`: the number of bins in a linear spectrogram
* `f_min`: the lowest frequency of the lowest band in a spectrogram
* `f_max`: the highest frequency of the highest band in a spectrogram
* `win_length`: the length of the STFT window
* `window_fn`: for functions that creates windows e.g. `torch.hann_window`

## License

By contributing to Torchaudio, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
