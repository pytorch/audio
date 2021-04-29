# Source Separation Example

This directory contains reference implementations for source separations. For the detail of each model, please checkout the followings.

- [Conv-TasNet](./conv_tasnet/README.md)

## Usage

### Overview

To training a model, you can use [`train.py`](./train.py). This script takes the form of
`train.py [parameters for distributed training] -- [parameters for model/training]`

    ```
    python train.py \
            [--worker-id WORKER_ID] \
            [--device-id DEVICE_ID] \
            [--num-workers NUM_WORKERS] \
            [--sync-protocol SYNC_PROTOCOL] \
            -- \
            <model specific training parameters>

    # For the detail of the parameter values, use;
    python train.py --help

    # For the detail of the model parameters, use;
    python train.py -- --help
    ```

If you would like to just try out the traing script, then try it without any parameters
for distributed training. `train.py -- --sample-rate 8000 --batch-size <BATCH_SIZE> --dataset-dir <DATASET_DIR> --save-dir <SAVE_DIR>`

This script runs training in Distributed Data Parallel (DDP) framework and has two major
operation modes. This behavior depends on if `--worker-id` argument is given or not.

1. (`--worker-id` is not given) Launchs training worker subprocesses that performs the actual training.
2. (`--worker-id` is given) Performs the training as a part of distributed training.

When launching the script without any distributed trainig parameters (operation mode 1),
this script will check the number of GPUs available on the local system and spawns the same
number of training subprocesses (as operaiton mode 2). You can reduce the number of GPUs with
`--num-workers`. If there is no GPU available, only one subprocess is launched and providing
`--num-workers` larger than 1 results in error.

When launching the script as a worker process of a distributed training, you need to configure
the coordination of the workers.

- `--num-workers` is the number of training processes being launched.
- `--worker-id` is the process rank (must be unique across all the processes).
- `--device-id` is the GPU device ID (should be unique within node).
- `--sync-protocl` is how each worker process communicate and synchronize.
  If the training is carried out on a single node, then the default `"env://"` should do.
  If the training processes span across multiple nodes, then you need to provide a protocol that
  can communicate over the network. If you know where the master node is located, you can use
  `"env://"` in combination with `MASTER_ADDR` and `MASER_PORT` environment variables. If you do
  not know where the master node is located beforehand, you can use `"file://..."` protocol to
  indicate where the file to which all the worker process have access is located. For other
  protocols, please refer to the official documentation.

### Distributed Training Notes

<details><summary>Quick overview on DDP (distributed data parallel)</summary>

DDP is single-program multiple-data training paradigm.
With DDP, the model is replicated on every process,
and every model replica will be fed with a different set of input data samples.

- **Process**: Worker process (as in Linux process). There are `P` processes per a Node.
- **Node**: A machine. There are `N` machines, each of which holds `P` processes.
- **World**: network of nodes, composed of `N` nodes and `N * P` processes.
- **Rank**: Grobal process ID (unique across nodes) `[0, N * P)`
- **Local Rank**: Local process ID (unique only within a node) `[0, P)`

```
          Node 0                    Node 1                          Node N-1
┌────────────────────────┐┌─────────────────────────┐     ┌───────────────────────────┐
│╔══════════╗ ┌─────────┐││┌───────────┐ ┌─────────┐│     │┌─────────────┐ ┌─────────┐│
│║ Process  ╟─┤ GPU: 0  ││││ Process   ├─┤ GPU: 0  ││     ││ Process     ├─┤ GPU: 0  ││
│║ Rank: 0  ║ └─────────┘│││ Rank:P    │ └─────────┘│     ││ Rank:NP-P   │ └─────────┘│
│╚══════════╝            ││└───────────┘            │     │└─────────────┘            │
│┌──────────┐ ┌─────────┐││┌───────────┐ ┌─────────┐│     │┌─────────────┐ ┌─────────┐│
││ Process  ├─┤ GPU: 1  ││││ Process   ├─┤ GPU: 1  ││     ││ Process     ├─┤ GPU: 1  ││
││ Rank: 1  │ └─────────┘│││ Rank:P+1  │ └─────────┘│     ││ Rank:NP-P+1 │ └─────────┘│
│└──────────┘            ││└───────────┘            │ ... │└─────────────┘            │
│                        ││                         │     │                           │
│ ...                    ││ ...                     │     │ ...                       │
│                        ││                         │     │                           │
│┌──────────┐ ┌─────────┐││┌───────────┐ ┌─────────┐│     │┌─────────────┐ ┌─────────┐│
││ Process  ├─┤ GPU:P-1 ││││ Process   ├─┤ GPU:P-1 ││     ││ Process     ├─┤ GPU:P-1 ││
││ Rank:P-1 │ └─────────┘│││ Rank:2P-1 │ └─────────┘│     ││ Rank:NP-1   │ └─────────┘│
│└──────────┘            ││└───────────┘            │     │└─────────────┘            │
└────────────────────────┘└─────────────────────────┘     └───────────────────────────┘
```

</details>

### SLURM

When launched as SLURM job, the follwoing environment variables correspond to

- **SLURM_PROCID*: `--worker-id` (Rank)
- **SLURM_NTASKS** (or legacy **SLURM_NPPROCS**): the number of total processes (`--num-workers` == world size)
- **SLURM_LOCALID**: Local Rank (to be mapped with GPU index*)

* Even when GPU resource is allocated with `--gpus-per-task=1`, if there are muptiple
tasks allocated on the same node, (thus multiple GPUs of the node are allocated to the job)
each task can see all the GPUs allocated for the tasks. Therefore we need to use
SLURM_LOCALID to tell each task to which GPU it should be using.

<details><summary>Example scripts for running the training on SLURM cluster</summary>

- **launch_job.sh**

```bash
#!/bin/bash

#SBATCH --job-name=source_separation

#SBATCH --output=/checkpoint/%u/jobs/%x/%j.out

#SBATCH --error=/checkpoint/%u/jobs/%x/%j.err

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=8

#SBATCH --mem-per-cpu=16G

#SBATCH --gpus-per-task=1

#srun env
srun wrapper.sh $@
```

- **wrapper.sh**

```bash
#!/bin/bash
num_speakers=2
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
save_dir="/checkpoint/${USER}/jobs/${SLURM_JOB_NAME}/${SLURM_JOB_ID}"
dataset_dir="/dataset/wsj0-mix/${num_speakers}speakers/wav8k/min"

if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
    protocol="file:///checkpoint/${USER}/jobs/source_separation/${SLURM_JOB_ID}/sync"
else
    protocol="env://"
fi

mkdir -p "${save_dir}"

python -u \
  "${this_dir}/train.py" \
  --worker-id "${SLURM_PROCID}" \
  --num-workers "${SLURM_NTASKS}" \
  --device-id "${SLURM_LOCALID}" \
  --sync-protocol "${protocol}" \
  -- \
  --num-speakers "${num_speakers}" \
  --sample-rate 8000 \
  --dataset-dir "${dataset_dir}" \
  --save-dir "${save_dir}" \
  --batch-size $((16 / SLURM_NTASKS))
```

</details>
