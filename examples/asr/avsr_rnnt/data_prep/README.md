
# Preprocessing LRS3

We provide a pre-processing pipeline to detect and crop full-face images in this repository.

## Prerequisites

Install all dependency-packages.

```Shell
pip install -r requirements.txt
```

Install [RetinaFace](./tools) tracker.

## Preprocessing

### Step 1. Pre-process the LRS3 dataset.
Please run the following script to pre-process the LRS3 dataset:

```Shell
python main.py \
    --data-dir=[data_dir] \
    --dataset=[dataset] \
    --root=[root] \
    --folder=[folder] \
    --groups=[num_groups] \
    --job-index=[job_index]
```

- `[data_dir]` and `[landmarks_dir]` are the directories for original dataset and corresponding landmarks.

- `[root]` is the directory for saved cropped-face dataset.

- `[folder]` can be set to  `train` or `test`.

- `[num_groups]` and `[job-index]` are used to split the dataset into multiple threads, where `[job-index]` is an integer in [0, `[num_groups]`).

### Step 2. Merge the label list.
After completing Step 2, run the following script to merge all labels.

```Shell
python merge.py \
    --dataset=[dataset] \
    --root=[root] \
    --folder=[folder] \
    --groups=[num_groups] \
```
