import contextlib

import torch


@contextlib.contextmanager
def use_deterministic_algorithms(mode: bool, warn_only: bool):
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield {}
    except RuntimeError as err:
        raise err
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
