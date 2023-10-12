#pragma once

#include <libtorio/ffmpeg/ffmpeg.h>

namespace torchaudio::io {

AVBufferRef* get_cuda_context(int index);

void clear_cuda_context_cache();

} // namespace torchaudio::io
