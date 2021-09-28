# Source Separation Example

This directory contains reference implementations for source separations. For the detail of each model, please checkout the followings.

- [Conv-TasNet](./conv_tasnet/README.md)

## Usage

### Overview

To training a model, you can use [`lightning_train.py`](./lightning_train.py). This script takes the form of
`ligntning_train.py [parameters]`

    ```
    python ligntning_train.py \
            [--data-dir DATA_DIR] \
            [--num-gpu NUM_GPU] \
            [--num-workers NUM_WORKERS] \
            ...

    # For the detail of the parameter values, use;
    python lightning_train.py --help
    ```

This script runs training in PyTorch-Lightning framework with Distributed Data Parallel (DDP) backend.
### SLURM

<details><summary>Example scripts for running the training on SLURM cluster</summary>

- **launch_job.sh**

```bash
#!/bin/bash

#SBATCH --job-name=source_separation

#SBATCH --output=/checkpoint/%u/jobs/%x/%j.out

#SBATCH --error=/checkpoint/%u/jobs/%x/%j.err

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=2

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=16G

#SBATCH --gpus-per-node=2

#srun env
srun wrapper.sh $@
```

- **wrapper.sh**

```bash
#!/bin/bash
num_speakers=2
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
exp_dir="/checkpoint/${USER}/exp/"
dataset_dir="/dataset/Libri${num_speakers}mix//wav8k/min"


mkdir -p "${exp_dir}"

python -u \
  "${this_dir}/ligntning_train.py" \
  --num-speakers "${num_speakers}" \
  --sample-rate 8000 \
  --data-dir "${dataset_dir}" \
  --exp-dir "${exp_dir}" \
  --batch-size $((16 / SLURM_NTASKS))
```

</details>
