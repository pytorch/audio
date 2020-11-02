# Conv-TasNet

This is a reference implementation of Conv-TasNet.

> Luo, Yi, and Nima Mesgarani. "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation." IEEE/ACM Transactions on Audio, Speech, and Language Processing 27.8 (2019): 1256-1266. Crossref. Web.

This implementation is based on [arXiv:1809.07454v3](https://arxiv.org/abs/1809.07454v3) and [the reference implementation](https://github.com/naplab/Conv-TasNet) provided by the authors.

For the usage, please checkout the [source separation README](../README.md).

## (Default) Training Configurations

The default training/model configurations follow the best non-causal implementation from the paper. (causal configuration is not implemented.)

 - Sample rate: 8000 Hz
 - Batch size: total 16 over distributed training workers
 - Epochs: 100
 - Initial learning rate: 1e-3
 - Gradient clipping: maximum L2 norm of 5.0
 - Optimizer: Adam
 - Learning rate scheduling: Halved after 3 epochs of no improvement in validation accuracy.
 - Objective function: SI-SNRi
 - Reported metrics: SI-SNRi, SDRi
 - Sample audio length: 4 seconds (randomized position)
 - Encoder/Decoder feature dimension (N): 512
 - Encoder/Decoder convolution kernel size (L): 16
 - TCN bottleneck/output feature dimension (B): 128
 - TCN hidden feature dimension (H): 512
 - TCN skip connection feature dimension (Sc): 128
 - TCN convolution kernel size (P): 3
 - The number of TCN convolution block layers (X): 8
 - The number of TCN convolution blocks (R): 3

## Evaluation

The following is the evaluation result of training the model on WSJ0-2mix and WSJ0-3mix datasets.

### wsj0-mix 2speakers

|                    | SI-SNRi (dB) | SDRi (dB) | Epoch |
|:------------------:|-------------:|----------:|------:|
| Reference          |         15.3 |      15.6 |       |
| Validation dataset |         13.1 |      13.1 |   100 |
| Evaluation dataset |         11.0 |      11.0 |   100 |

### wsj0-mix 3speakers

|                    | SI-SNRi (dB) | SDRi (dB) | Epoch |
|:------------------:|-------------:|----------:|------:|
| Reference          |         12.7 |      13.1 |       |
| Validation dataset |         11.4 |      11.4 |   100 |
| Evaluation dataset |          8.9 |       8.9 |   100 |
