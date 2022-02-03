# Augmentation

This example demonstrates how you can use torchaudio's I/O features and augmentations in C++ application.

**NOTE**
This example uses `"sox_io"` backend, thus does not work on Windows.

## Steps
### 1. Create augmentation pipeline TorchScript file.

First, we implement our data process pipeline as a regular Python, and save it as a TorchScript object.
We will load and execute it in our C++ application. The C++ code is found in [`main.cpp`](./main.cpp).

```python
python create_jittable_pipeline.py \
    --rir-path "../data/rir.wav" \
    --output-path "./pipeline.zip"
```

### 2. Build the application

Please refer to [the top level README.md](../README.md)

### 3. Run the application

Now we run the C++ application `augment`, with the TorchScript object we created in Step.1 and an input audio file.

In [the top level directory](../)

```bash
input_audio_file="./data/input.wav"
./build/augmentation/augment ./augmentation/pipeline.zip "${input_audio_file}" "output.wav"
```

When you give a clean speech file, the output audio sounds like it's a phone conversation.
