import warnings
import importlib.util
from functools import wraps


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


def depricate(direction: str):
    """Decorator to add depciration message"""
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(f'{func.__module__}.{func.__name__} has been deprecated. f{direction}')
            return func(*args, **kwargs)
        return wrapped
    return decorator
