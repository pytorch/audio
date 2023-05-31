#include <ATen/DynamicLibrary.h>
#include <c10/util/CallOnce.h>
#include <torchaudio/csrc/sox/libsox.h>
#include <torchaudio/csrc/sox/utils.h>

#include <memory>

#include <cstdlib>

namespace torchaudio::sox {
namespace {

// Handle to the dlopen-ed libsox
class LSXImpl {
  at::DynamicLibrary handle;

 public:
  LSX lsx;

  LSXImpl(const char* name) : handle(name) {
    // check version: we only support 14.4.2
    {
      auto version = ((const char* (*)(void))handle.sym("sox_version"))();
      TORCH_CHECK(
          strcmp(version, "14.4.2") == 0,
          "Need libsox 14.4.2, but found ",
          version);
    }

    // Register fanction pointers on public-facing interface
#define set_func(NAME) this->lsx.NAME = (decltype(LSX::NAME))handle.sym(#NAME)
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
    set_func(sox_open_read);
    set_func(sox_open_write);
    set_func(sox_strerror);
    set_func(sox_write);
#undef set_func

    // Init sox effect plugins
    auto fn = (int (*)())handle.sym("sox_init");
    TORCH_CHECK(SOX_SUCCESS == fn(), "Failed to initialize sox effects.");
  }

  ~LSXImpl() {
    auto fn = (int (*)())handle.sym("sox_quit");
    if (SOX_SUCCESS != fn()) {
      TORCH_WARN("Failed to release sox effect plugins.");
    }
  }
};

static std::unique_ptr<LSXImpl> libsox;

} // namespace

// Fetch lsx
#if defined(_WIN32)
#define EXT ".lib"
#elif defined(__APPLE__)
#define EXT ".dylib"
#else
#define EXT ".so"
#endif
LSX& lsx() {
  static c10::once_flag init_flag;
  c10::call_once(init_flag, [](){
    libsox.reset(new LSXImpl("libsox" EXT));
    auto config = libsox->lsx.sox_get_globals();
    config->verbosity = 0;
    config->use_threads = sox_false;
  });
  static LSX& ret = libsox->lsx;
  return ret;
}
#undef EXT

} // namespace torchaudio::sox
