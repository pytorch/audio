# Custom Kaldi build

This directory contains original Kaldi repository (as submodule), [the custom implementation of Kaldi's vector/matrix](./src) and the build script.

We use the custom build process so that the resulting library only contains what torchaudio needs.
We use the custom vector/matrix implementation so that we can use the same BLAS library that PyTorch is compiled with, and so that we can (hopefully, in future) take advantage of other PyTorch features (such as differentiability and GPU support). The down side of this approach is that it adds a lot of overhead compared to the original Kaldi (operator dispatch and element-wise processing, which PyTorch is not efficient at). We can improve this gradually, and if you are interested in helping, please let us know by opening an issue.