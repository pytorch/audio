def _init_fb_ctypes():
    # Initiaization required only in facebook infrastructure to use soundfile
    import libfb.py.ctypesmonkeypatch
    libfb.py.ctypesmonkeypatch.install()


try:
    _init_fb_ctypes()
except Exception:
    pass
