#!/usr/bin/env python3
"""Launch souce separation training.

This script runs training in Distributed Data Parallel (DDP) framework and has two major
operation modes. This behavior depends on if `--worker-id` argument is given or not.

1. (`--worker-id` is not given) Launchs worker subprocesses that performs the actual training.
2. (`--worker-id` is given) Performs the training as a part of distributed training.

When launching the script without any distributed trainig parameters (operation mode 1),
this script will check the number of GPUs available on the local system and spawns the same
number of training subprocesses (as operaiton mode 2). You can reduce the number of GPUs with
`--num-workers`. If there is no GPU available, only one subprocess is launched.

When launching the script as a worker process of a distributed training, you need to configure
the coordination of the workers.
"""
import sys
import logging
import argparse
import subprocess

import torch

from utils import dist_utils

_LG = dist_utils.getLogger(__name__)


def _parse_args(args=None):
    max_world_size = torch.cuda.device_count() or 1

    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    group = parser.add_argument_group("Distributed Training")
    group.add_argument(
        "--worker-id",
        type=int,
        help=(
            "If not provided, the launched process serves as a master process of "
            "single-node, multi-worker training and spawns the worker subprocesses. "
            "If provided, the launched process serves as a worker process, which "
            "performs the actual training. The valid value is [0, --num-workers)."
        ),
    )
    group.add_argument(
        "--device-id",
        type=int,
        help="The CUDA device ID. Allowed only when --worker-id is provided.",
    )
    group.add_argument(
        "--num-workers",
        type=int,
        default=max_world_size,
        help=(
            "The size of distributed trainig workers. "
            "If launching a training as single-node, multi-worker training, "
            "(i.e. --worker-id is not provided) then this value should not exceed "
            "the number of available GPUs. "
            "If launching the training process as a multi-node, multi-gpu training, "
            "(i.e. --worker-id is provided) then the value has to match "
            f"the number of workers across nodes. (default: {max_world_size})"
        ),
    )
    group.add_argument(
        "--sync-protocol",
        type=str,
        default="env://",
        help=(
            "Synchronization protocol for distributed training. "
            "This value is passed as `init_method` argument of "
            "`torch.distributed.init_process_group` function."
            'If you are using `"env://"`, you can additionally configure '
            'environment variables "MASTER_ADDR" and "MASTER_PORT". '
            'If you are using `"file://..."`, then the process has to have '
            "the access to the designated file. "
            "See the documentation for `torch.distributed` for the detail. "
            'If you are running the training in a single node, `"env://"` '
            "should do. If you are running the training in multiple nodes, "
            "you need to provide the file location where all the nodes have "
            'access, using `"file://..."` protocol. (default: "env://")'
        ),
    )
    group.add_argument(
        "--random-seed",
        type=int,
        help="Set random seed value. (default: None)",
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER, help="Model-specific arguments."
    )
    namespace = parser.parse_args(args)
    if namespace.worker_id is None:
        if namespace.device_id is not None:
            raise ValueError(
                "`--device-id` cannot be provided when runing as master process."
            )
        if namespace.num_workers > max_world_size:
            raise ValueError(
                "--num-workers ({num_workers}) cannot exceed {device_count}."
            )
    if namespace.rest[:1] == ["--"]:
        namespace.rest = namespace.rest[1:]
    return namespace


def _main(cli_args):
    args = _parse_args(cli_args)

    if any(arg in ["--help", "-h"] for arg in args.rest):
        _run_training(args.rest)

    _init_logger(args.worker_id, args.debug)
    if args.worker_id is None:
        _run_training_subprocesses(args.num_workers, cli_args)
    else:
        dist_utils.setup_distributed(
            world_size=args.num_workers,
            rank=args.worker_id,
            local_rank=args.device_id,
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method=args.sync_protocol,
        )
        if args.random_seed is not None:
            torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.device_id)
            _LG.info("CUDA device set to %s", args.device_id)
        _run_training(args.rest)


def _run_training_subprocesses(num_workers, original_args):
    workers = []
    _LG.info("Spawning %s workers", num_workers)
    for i in range(num_workers):
        worker_arg = ["--worker-id", f"{i}", "--num-workers", f"{num_workers}"]
        device_arg = ["--device-id", f"{i}"] if torch.cuda.is_available() else []
        command = (
            [sys.executable, "-u", sys.argv[0]]
            + worker_arg
            + device_arg
            + original_args
        )
        _LG.info("Launching worker %s: `%s`", i, " ".join(command))
        worker = subprocess.Popen(command)
        workers.append(worker)

    num_failed = 0
    for worker in workers:
        worker.wait()
        if worker.returncode != 0:
            num_failed += 1
    sys.exit(num_failed)


def _run_training(args):
    import conv_tasnet.train

    conv_tasnet.train.train(args)


def _init_logger(rank=None, debug=False):
    worker_fmt = "[master]" if rank is None else f"[worker {rank:2d}]"
    message_fmt = (
        "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    )
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {worker_fmt} {message_fmt}",
    )


if __name__ == "__main__":
    _main(sys.argv[1:])
