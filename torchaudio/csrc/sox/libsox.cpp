#include <torchaudio/csrc/sox/libsox.h>
#include <c10/util/CallOnce.h>

namespace torchaudio::sox {
namespace {

// Handle to the dlopen-ed libsox
static std::unique_ptr<at::DynamicLibrary> libsox;
// LSX class which torchaudio will be using
static LSX _lsx;
// dlopen libsox and populate methods on _lsx,
// then perform initialization.
void _init_lsx();

} // namespace

// Fetch lsx
LSX& lsx() {
  static c10::once_flag init_flag;
  c10::call_once(init_flag, _init_lsx);
  return _lsx;
}

namespace {

// dlopen libsox and populate methods on _lsx.
void _init_lsx() {
  libsox = []() {
#if defined(_WIN32)
#error Windows is not supported.
#elif defined(__APPLE__)
    auto lsx_ =
        std::make_unique<at::DynamicLibrary>("libsox.3.dylib", "libsox.dylib");
#else
    auto lsx_ =
        std::make_unique<at::DynamicLibrary>("libsox.3.so", "libsox.so");
#endif
    // check version: we only support 14.4.2
    auto fn = (const char* (*)(void))lsx_->sym("sox_version");
    TORCH_CHECK(
        strcmp(fn(), "14.4.2") == 0,
        "Need libsox 14.4.2, but found", fn());
    return lsx_;
  }();

#define set_func(NAME) _lsx.NAME = (decltype(LSX::NAME))libsox->sym(#NAME)

  // Note
  // If any of the following fails, it will leave _lsx in invalid state.
  // But _lsx cannot be accessed without this fuction succeful, so it'okay.
  set_func(sox_add_effect);
  set_func(sox_close);
  set_func(sox_create_effect);
  set_func(sox_create_effects_chain);
  set_func(sox_delete_effect);
  set_func(sox_delete_effects_chain);
  set_func(sox_effect_options);
  set_func(sox_find_effect);
  set_func(sox_flow_effects);
  set_func(sox_get_effect_fns);
  set_func(sox_get_format_fns);
  set_func(sox_get_globals);
  set_func(sox_init);
  set_func(sox_open_read);
  set_func(sox_open_write);
  set_func(sox_quit);
  set_func(sox_strerror);
  set_func(sox_write);
#undef set_func
}

} // namespace
} // namespace torchaudio::sox
