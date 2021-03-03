# Example usage: libtorchaudio

This example demonstrates how you can use torchaudio's I/O features in C++ application, in addition to PyTorch's operations.

To try this example, simply run `./run.sh`. This script will

1. Cretae an audio preprocessing pipeline with TorchScript and dump it to the file.
2. Build the application using `libtorch` and `libtorchaudio`.
3. Execute the preprocessing pipeline on the example audio.

The detail of the preprocessing pipeline can be found in [`create_jittable_pipeline.py`](./create_jittable_pipeline.py).
