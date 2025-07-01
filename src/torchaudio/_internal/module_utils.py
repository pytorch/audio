import importlib.util
import os
import warnings
from functools import wraps, partial
from typing import Optional


def eval_env(var, default):
    """Check if environment varable has True-y value"""
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        # fmt: off
        raise RuntimeError(
            f"Unexpected environment variable value `{var}={val}`. "
            f"Expected one of {trues + falses}")
        # fmt: on
    return False


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def requires_module(*modules: str):
    """Decorate function to give error message if invoked without required optional modules.

    This decorator is to give better error message to users rather
    than raising ``NameError:  name 'module' is not defined`` at random places.
    """
    missing = [m for m in modules if not is_module_available(m)]

    if not missing:
        # fall through. If all the modules are available, no need to decorate
        def decorator(func):
            return func

    else:
        req = f"module: {missing[0]}" if len(missing) == 1 else f"modules: {missing}"

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f"{func.__module__}.{func.__name__} requires {req}")

            return wrapped

    return decorator

UNSUPPORTED = []

def wrap_deprecated(func, name, direction: str, version: Optional[str] = None, remove: bool = False):
    @wraps(func)
    def wrapped(*args, **kwargs):
        message = f"{name} has been deprecated. {direction}"
        if remove:
            message += f' It will be removed from {"a future" if version is None else "the " + str(version)} release. '
        warnings.warn(message, stacklevel=2)
        return func(*args, **kwargs)
    return wrapped

def deprecated(direction: str, version: Optional[str] = None, remove: bool = False):
    """Decorator to add deprecation message

    Args:
        direction (str): Migration steps to be given to users.
        version (str or int): The version when the object will be removed
        remove (bool): If enabled, append future removal message.
    """

    def decorator(func):
        wrapped = wrap_deprecated(func, f"{func.__module__}.{func.__name__}", direction, version=version, remove=remove)

        message = "This function has been deprecated. "
        if remove:
            message += f'It will be removed from {"future" if version is None else version} release. '

        wrapped.__doc__ = f"""DEPRECATED: {func.__doc__}

    .. warning::

       {message}
       {direction}
        """

        UNSUPPORTED.append(wrapped)
        return wrapped

    return decorator

DEPRECATION_MSG = (
    "As TorchAudio is no longer being actively developed, this functionality can no longer be supported. "
    "See https://github.com/pytorch/audio/issues/3902 for more details.")

IO_DEPRECATION_MSG = (
    "This functionality has been superseded by `AudioDecoder` from the TorchCodec library. "
    "See https://github.com/pytorch/audio/issues/3902 for more details.")

dropping_support = deprecated(DEPRECATION_MSG, version="2.9", remove=True)

def dropping_class_support(c, msg=DEPRECATION_MSG):
    c.__init__ = wrap_deprecated(c.__init__, f"{c.__module__}.{c.__name__}", msg, version="2.9", remove=True)
    c.__doc__ = f"""DEPRECATED: {c.__doc__}

.. warning::

    This class has been deprecated. It will be removed from the 2.9 release.
    {msg}
    """

    UNSUPPORTED.append(c)
    return c

def dropping_const_support(c, msg=DEPRECATION_MSG, name=None):
    c.__forward__ = wrap_deprecated(c.__init__, name or f"{c.__module__}.{c.__name__}", msg, version="2.9", remove=True)
    c.__doc__ = f"""DEPRECATED: {c.__doc__}

.. warning::

    This object has been deprecated. It will be removed from the 2.9 release.
    {msg}
    """

    UNSUPPORTED.append(c)
    return c
dropping_class_io_support = partial(dropping_class_support, msg=IO_DEPRECATION_MSG)

dropping_io_support = deprecated(IO_DEPRECATION_MSG, version="2.9", remove=True)

def fail_with_message(message):
    """Generate decorator to give users message about missing TorchAudio extension."""

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            raise RuntimeError(f"{func.__module__}.{func.__name__} {message}")

        return wrapped

    return decorator


def no_op(func):
    """Op-op decorator. Used in place of fail_with_message when a functionality that requires extension works fine."""
    return func
