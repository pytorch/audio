
# Pre-process LRS3

We provide a pre-processing pipeline in this repository for detecting and cropping full-face regions of interest (ROIs) as well as corresponding audio waveforms for LRS3.

## Introduction

Before feeding the raw stream into our model, each video sequence has to undergo a specific pre-processing procedure. This involves three critical steps. The first step is to perform face detection. Following that, each individual frame is aligned to a referenced frame, commonly known as the mean face, in order to normalize rotation and size differences across frames. The final step in the pre-processing module is to crop the face region from the aligned face image.

<div align="center">

<table style="display: inline-table;">
<tr><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/original.gif", width="144"></td><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/detected.gif" width="144"></td><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/transformed.gif" width="144"></td><td><img src="https://download.pytorch.org/torchaudio/doc-assets/avsr/cropped.gif" width="144"></td></tr>
<tr><td>0. Original</td> <td>1. Detection</td> <td>2. Transformation</td> <td>3. Face ROIs</td> </tr>
</table>
</div>

## Preparation

1. Install all dependency-packages.

```Shell
pip install -r requirements.txt
```

2. Install [retinaface](./tools) or [mediapipe](https://pypi.org/project/mediapipe/) tracker. If you have installed the tracker, please skip it.

## Preprocessing LRS3

To pre-process the LRS3 dataset, plrase follow these steps:

1. Download the LRS3 dataset from the official website.

2. Run the following command to preprocess the dataset:

```Shell
python preprocess_lrs3.py \
    --data-dir=[data_dir] \
    --detector=[detector] \
    --dataset=[dataset] \
    --root-dir=[root] \
    --subset=[subset] \
    --seg-duration=[seg_duration] \
    --groups=[n] \
    --job-index=[j]
```

- `data-dir`: Path to the directory containing video files.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `dataset`: Name of the dataset. Valid value is: `lrs3`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `subset`: Name of the subset. Valid values are: `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `16`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group. Valid values are an integer within the range of `[0, n)`.

3. Run the following command to merge all labels:

```Shell
python merge.py \
    --root-dir=[root_dir] \
    --dataset=[dataset] \
    --subset=[subset] \
    --seg-duration=[seg_duration] \
    --groups=[n]
```

- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of the dataset. Valid values are: `lrs2` and `lrs3`.
- `subset`: The subset name of the dataset. For LRS2, valid values are `train`, `val`, and `test`. For LRS3, valid values are `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `16`.
- `groups`: Number of groups to split the dataset into.
