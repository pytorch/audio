This is an example pipeline for text-to-speech using Tacotron2.


## Install required packages

Required packages
```bash
pip install librosa tqdm inflect
```

To use tensorboard
```bash
pip install tensorboard pillow
```

## Training Tacotron2

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
    --text-preprocessor character \
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