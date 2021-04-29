# Torchscript Backward Compatibility Test Suite

This directory contains tools to generate Torchscript object of a specific torchaudio version (given that you have the corresponding environments setup correctly) and validate it in the current version.

## Usage

### Generate torchscript object

```
./main.py --mode generate --version 0.6.0
```

will generate Torchscript dump files in [`assets`](./assets/) directory. This requries your Python runtime to have torchaudio `0.6.0`.


### Validate torchscript object


```
./main.py --mode validate --version 0.6.0
```

will validate if the Torchscript files created in the previous step are compatible with the version of torchaudio available in your environment (master).
