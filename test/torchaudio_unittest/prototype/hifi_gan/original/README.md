# Reference Implementation of HiFiGAN

The code in this folder was taken from the original implementation
https://github.com/jik876/hifi-gan/tree/4769534d45265d52a904b850da5a622601885777
which was made available the following liscence:

MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code is used for testing that our implementation matches the original one. To enable such testing the
ported code has been are modified in a minimal way, namely:
 - Remove objects other than `mel_spectrogram` and its dependencies from `meldataset.py`
 - Remove objects other than `AttrDict` from `env.py`
 - Remove objects other than `init_weights` and `get_padding` from `utils.py`
 - Add `return_complex=False` argument to `torch.stft` call in `mel_spectrogram` in `meldataset.py`, to make code
PyTorch 2.0 compatible
 - Remove the import statements required only for the removed functions.
  - Add `# flake8: noqa` at the top of each file so as not to report any format issue on the ported code.
The implementation of the retained functions and classes and their formats are kept as-is.
