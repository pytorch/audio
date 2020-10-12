import warnings
import importlib.util
from typing import Optional
from functools import wraps, WRAPPER_ASSIGNMENTS


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
        req = f'module: {missing[0]}' if len(missing) == 1 else f'modules: {missing}'

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f'{func.__module__}.{func.__name__} requires {req}')
            return wrapped
    return decorator


def deprecated(direction: str, version: Optional[str] = None):
    """Decorator to add deprecation message

    Args:
        direction: Migration steps to be given to users.
    """
    def decorator(obj):
        message = (
            f'{obj.__module__}.{obj.__name__} has been deprecated and will be '
            f'removed from {"future" if version is None else version} release. '
            f'{direction}')
        if isinstance(obj, type):
            return _decorate_class(obj, message)
        else:
            return _decorate_func(obj, message)
    return decorator

def _decorate_class(obj, message):
    # Update the class to emit a warning when used
    class Wrapper(obj):
        def __init__(self, *args, **kwargs):
            warnings.warn(message, stacklevel=1)
            self._wrapped = obj
            for attr in WRAPPER_ASSIGNMENTS:
                setattr(self, attr, getattr(wrapped, attr))
            super(self).__init__(*args, **kwargs)

        def __repr__(self):
                    return repr(self._wrapped)

    return Wrapper

def _decorate_func(obj, message):
    @wraps(obj)
    def wrapped(*args, **kwargs):
        warnings.warn(message, stacklevel=2)
        return obj(*args, **kwargs)
    return wrapped
