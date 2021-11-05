# Source Separation Example

This directory contains reference implementations for source separations. For the detail of each model, please checkout the followings.

- [Conv-TasNet](./conv_tasnet/README.md)

## Usage

### Overview

To training a model, you can use [`lightning_train.py`](./lightning_train.py). This script takes the form of
`lightning_train.py [parameters]`

    ```
    python lightning_train.py \
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
root_dir="/dataset/" # The directory where the directory ``Libri2Mix`` or ``Libri3Mix`` is stored.
num_gpu=2 # The number of GPUs used on one node.
num_node=1 # The number of nodes used on the cluster.
batch_size=6 # The batch size per GPU.


mkdir -p "${exp_dir}"

python -u \
  "${this_dir}/lightning_train.py" \
  --num-speakers "${num_speakers}" \
  --sample-rate 8000 \
  --root-dir "${root_dir}" \
  --exp-dir "${exp_dir}" \
  --num-gpu ${num_gpu} \
  --num-node ${num_node} \
  --batch-size ${batch_size} \
```

</details>
