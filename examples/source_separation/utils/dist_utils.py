import os
import csv
import types
import logging

import torch
import torch.distributed as dist


def _info_on_master(self, *args, **kwargs):
    if dist.get_rank() == 0:
        self.info(*args, **kwargs)


def getLogger(name):
    """Get logging.Logger module with additional ``info_on_master`` method."""
    logger = logging.getLogger(name)
    logger.info_on_master = types.MethodType(_info_on_master, logger)
    return logger


_LG = getLogger(__name__)


def setup_distributed(
    world_size, rank, local_rank, backend="nccl", init_method="env://"
):
    """Perform env setup and initialization for distributed training"""
    if init_method == "env://":
        _set_env_vars(world_size, rank, local_rank)
    if world_size > 1 and "OMP_NUM_THREADS" not in os.environ:
        _LG.info("Setting OMP_NUM_THREADS == 1")
        os.environ["OMP_NUM_THREADS"] = "1"
    params = {
        "backend": backend,
        "init_method": init_method,
        "world_size": world_size,
        "rank": rank,
    }
    _LG.info("Initializing distributed process group with %s", params)
    dist.init_process_group(**params)
    _LG.info("Initialized distributed process group.")


def _set_env_vars(world_size, rank, local_rank):
    for key, default in [("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29500")]:
        if key not in os.environ:
            os.environ[key] = default

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)


def save_on_master(path, obj):
    if dist.get_rank() == 0:
        _LG.info("Saving %s", path)
        torch.save(obj, path)


def write_csv_on_master(path, *rows):
    if dist.get_rank() == 0:
        with open(path, "a", newline="") as fileobj:
            writer = csv.writer(fileobj)
            for row in rows:
                writer.writerow(row)


def synchronize_params(path, device, *modules):
    if dist.get_world_size() < 2:
        return
    rank = dist.get_rank()
    if rank == 0:
        _LG.info("[Parameter Sync]: Saving parameters to a temp file...")
        torch.save({f"{i}": m.state_dict() for i, m in enumerate(modules)}, path)
    dist.barrier()
    if rank != 0:
        _LG.info("[Parameter Sync]:    Loading parameters...")
        data = torch.load(path, map_location=device)
        for i, m in enumerate(modules):
            m.load_state_dict(data[f"{i}"])
    dist.barrier()
    if rank == 0:
        _LG.info("[Parameter Sync]: Removing the temp file...")
        os.remove(path)
    _LG.info_on_master("[Parameter Sync]: Complete.")
