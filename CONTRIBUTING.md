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

## Development installation

We recommend using a `conda` environment to contribute efficiently to
torchaudio.

### Install PyTorch Nightly 

```bash
conda install pytorch -c pytorch-nightly
```

### Install Torchaudio

```bash
pip install cmake ninja
```

```bash
git clone https://github.com/pytorch/audio.git
cd audio
git submodule update --init --recursive
BUILD_SOX=1 python setup.py develop
# or, for OSX
# BUILD_SOX=1 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py develop
# for C++ debugging, please use DEBUG=1
# DEBUG=1 python setup.py develop
```

Note: you don't need to use `BUILD_SOX=1` if you have `libsox-dev` installed
already. If you built sox however, set the `PATH` variable so that the tests
properly use the newly built `sox` binary:

```bash
export PATH="<path_to_torchaudio>/third_party/install/bin:${PATH}"
```

The following dependencies are also needed for testing:

```bash
pip install typing pytest scipy numpy parameterized
```

## Development Process

If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `master`: `$ git checkout master && git checkout -b my_cool_feature`
2. If you have modified the code (new feature or bug-fix), [please add tests](.test/torchaudio_unittest/).
3. If you have changed APIs, [update the documentation](#Documentation).

For more details about pull requests, 
please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request). 

If you would like to contribute a new model, please see [here](#New-model).

If you would like to contribute a new dataset, please see [here](#New-dataset). 

## Testing

Please refer to our [testing guidelines](.test/torchaudio_unittest/) for more
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

The built docs should now be available in `docs/build/html`

## License

By contributing to Torchaudio, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
