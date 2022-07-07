#pragma once

// Define the visibility of symbols.
// The original logic and background can be found here.
// https://github.com/pytorch/pytorch/blob/bcc02769bef1d7b89bec724223284958b7c5b564/c10/macros/Export.h#L49-L55
//
// In the context of torchaudio, the logic is simpler at the moment.
//
// The torchaudio custom operations are implemented in
// `torchaudio/lib/libtorchaudio.[so|pyd]`. Some symbols are referred from
// `torchaudio._torchaudio`.
//
// In Windows, default visibility of dynamically library are hidden, while in
// Linux/macOS, they are visible.
//
// At the moment we do not expect torchaudio libraries to be built/linked
// statically. We assume they are always shared.

#ifdef _WIN32
#define TORCHAUDIO_EXPORT __declspec(dllexport)
#define TORCHAUDIO_IMPORT __declspec(dllimport)
#else // _WIN32
#if defined(__GNUC__)
#define TORCHAUDIO_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define TORCHAUDIO_EXPORT
#endif // defined(__GNUC__)
#define TORCHAUDIO_IMPORT TORCHAUDIO_EXPORT
#endif // _WIN32

#ifdef TORCHAUDIO_BUILD_MAIN_LIB
#define TORCHAUDIO_API TORCHAUDIO_EXPORT
#else
#define TORCHAUDIO_API TORCHAUDIO_IMPORT
#endif
