This is an example pipeline for text-to-speech using Tacotron2.


## Install required packages

Required packages
```bash
pip install librosa tqdm inflect joblib
```

To use tensorboard
```bash
pip install tensorboard pillow
```

## Training Tacotron2 with character as input

The training of Tacotron2 can be invoked with the following command.

```bash
python train.py \
    --learning-rate 1e-3 \
    --epochs 1501 \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 \
    --batch-size 96 \
    --weight-decay 1e-6 \
    --grad-clip 1.0 \
    --text-preprocessor english_characters \
    --logging-dir ./logs \
    --checkpoint-path ./ckpt.pth \
    --dataset-path ./
```

The training script will use all GPUs that is available, please set the
environment variable `CUDA_VISIBLE_DEVICES` if you don't want all GPUs to be used.
The newest checkpoint will be saved to `./ckpt.pth` and the checkpoint with the best validation
loss will be saved to `./best_ckpt.pth`.
The training log will be saved to `./logs/train.log` and the tensorboard results will also
be in `./logs`.

If `./ckpt.pth` already exist, this script will automatically load the file and try to continue
training from the checkpoint.

This command takes around 36 hours to train on 8 NVIDIA Tesla V100 GPUs.

To train the Tacotron2 model to work with the [pretrained wavernn](https://pytorch.org/audio/main/models.html#id10)
with checkpoint_name `"wavernn_10k_epochs_8bits_ljspeech"`, please run the following command instead.

```bash
python train.py
    --learning-rate 1e-3 \
    --epochs 1501 \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 \
    --sample-rate 22050 \
    --n-fft 2048 \
    --hop-length 275 \
    --win-length 1100 \
    --mel-fmin 40 \
    --mel-fmax 11025 \
    --batch-size 96 \
    --weight-decay 1e-6 \
    --grad-clip 1.0 \
    --text-preprocessor english_characters \
    --logging-dir ./wavernn_logs \
    --checkpoint-path ./ckpt_wavernn.pth \
    --dataset-path ./
```


## Training Tacotron2 with phoneme as input

#### Dependencies

This example use the [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) as
the phonemizer (the function to turn text into phonemes),
please install it with the following command (the code is tested with version 0.0.15).

```bash
pip install deep-phonemizer==0.0.15
```

Then download the model weights from [their website](https://github.com/as-ideas/DeepPhonemizer)

The link to the checkpoint that is tested with this example is
[https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt).

#### Running training script

The training of Tacotron2 with english phonemes as input can be invoked with the following command.

```bash
python train.py \
    --workers 12 \
    --learning-rate 1e-3 \
    --epochs 1501 \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 \
    --batch-size 96 \
    --weight-decay 1e-6 \
    --grad-clip 1.0 \
    --text-preprocessor english_phonemes \
    --phonemizer DeepPhonemizer \
    --phonemizer-checkpoint ./en_us_cmudict_forward.pt \
    --cmudict-root ./ \
    --logging-dir ./english_phonemes_logs \
    --checkpoint-path ./english_phonemes_ckpt.pth \
    --dataset-path ./
```

Similar to the previous examples, this command will save the log in the directory `./english_phonemes_logs`
and the checkpoint will be saved to `./english_phonemes_ckpt.pth`.


To train the Tacotron2 model with english phonemes that works with the
[pretrained wavernn](https://pytorch.org/audio/main/models.html#id10)
with checkpoint_name `"wavernn_10k_epochs_8bits_ljspeech"`, please run the following command.

```bash
python train.py \
    --workers 12 \
    --learning-rate 1e-3 \
    --epochs 1501 \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 \
    --sample-rate 22050 \
    --n-fft 2048 \
    --hop-length 275 \
    --win-length 1100 \
    --mel-fmin 40 \
    --mel-fmax 11025 \
    --batch-size 96 \
    --weight-decay 1e-6 \
    --grad-clip 1.0 \
    --text-preprocessor english_phonemes \
    --phonemizer DeepPhonemizer \
    --phonemizer-checkpoint ./en_us_cmudict_forward.pt \
    --cmudict-root ./ \
    --logging-dir ./english_phonemes_wavernn_logs \
    --checkpoint-path ./english_phonemes_wavernn_ckpt.pth \
    --dataset-path ./
```
