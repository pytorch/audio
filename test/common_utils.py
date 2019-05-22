import sys

PY3 = sys.version_info > (3, 0)
PY34 = sys.version_info >= (3, 4)


def _check_module_exists(name):
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    if not PY3:  # Python 2
        import imp
        try:
            imp.find_module(name)
            return True
        except ImportError:
            return False
    elif not PY34:  # Python [3, 3.4)
        import importlib
        loader = importlib.find_loader(name)
        return loader is not None
    else:  # Python >= 3.4
        import importlib
        import importlib.util
        spec = importlib.util.find_spec(name)
        return spec is not None


TEST_SCIPY = _check_module_exists('scipy')

# On Py2, importing librosa 0.6.1 triggers a TypeError (if using newest joblib)
# see librosa/librosa#729.
# TODO: allow Py2 when librosa 0.6.2 releases
TEST_LIBROSA = _check_module_exists('librosa') and PY3
