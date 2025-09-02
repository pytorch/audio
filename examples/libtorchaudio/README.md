# Libtorchaudio Examples

* [Speech Recognition with wav2vec2.0](./speech_recognition)

## Build

The example applications in this directory depend on `libtorch` and `libtorchaudio`.
If you have a working `PyTorch`, you already have `libtorch`.
Please refer to [this tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) for the use of `libtorch` and TorchScript.

`libtorchaudio` is the library of torchaudio's C++ components without Python component.
It is currently not distributed, and it will be built alongside with the applications.

The following commands will build `libtorchaudio` and applications.

```bash
git submodule update
mkdir build
cd build
cmake -GNinja \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      -DBUILD_SOX=ON \
      -DBUILD_KALDI=OFF \
      -DBUILD_RNNT=ON \
      ..
cmake --build .
```

For the usages of each application, refer to the corresponding application directory.
