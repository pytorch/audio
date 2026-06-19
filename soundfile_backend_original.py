    """
    with soundfile.SoundFile(filepath, "r") as file_:
        if file_.format != "WAV" or normalize:
            # When reading non-WAV formats (e.g. MP3) with normalize=True,
            # use the native floating-point dtype for the subtype if available,
            # falling back to float32. This ensures correct reading of files
            # that require float64 precision (e.g. certain MP3 files with
            # DOUBLE subtype). For integer subtypes, float32 is used for
            # normalization.
            subtype_dtype = _SUBTYPE2DTYPE.get(file_.subtype)
            if subtype_dtype in ("float64",):
                dtype = "float64"
            else:
                dtype = "float32"
        elif file_.subtype not in _SUBTYPE2DTYPE:
            raise ValueError(f"Unsupported subtype: {file_.subtype}")
        else:
