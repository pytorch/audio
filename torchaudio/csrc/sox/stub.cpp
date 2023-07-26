#include <ATen/DynamicLibrary.h>
#include <c10/util/CallOnce.h>
#include <torchaudio/csrc/sox/stub.h>
#include <memory>

namespace torchaudio::sox::detail {
namespace {

const char* get_compile_version() {
  static char versionstr[20];

  sprintf(
      versionstr,
      "%d.%d.%d",
      (SOX_LIB_VERSION_CODE & 0xff0000) >> 16,
      (SOX_LIB_VERSION_CODE & 0x00ff00) >> 8,
      (SOX_LIB_VERSION_CODE & 0x0000ff));
  return versionstr;
}

LibSoxStub get_stub(at::DynamicLibrary& handle) {
  // Validate version: It's only tested on 14.4.2, and we don't expect
  // new version of sox to be compatible nor we intend to support.
  auto fn = (const sox_version_info_t* (*)())handle.sym("sox_version_info");
  if (SOX_LIB_VERSION_CODE != fn()->version_code) {
    auto runtime_ver = ((const char* (*)())handle.sym("sox_version"))();
    TORCH_WARN(
        "TorchAudio was compiled with sox version ",
        get_compile_version(),
        ". But the version found is ",
        runtime_ver,
        ". If this causes a problem, you can disable sox integration by setting environment variable TORCHAUDIO_USE_SOX=0.");
  }

#define get_symbol(X) (decltype(LibSoxStub::X)) handle.sym(#X)
  return LibSoxStub{
      get_symbol(sox_add_effect),
      get_symbol(sox_close),
      get_symbol(sox_create_effect),
      get_symbol(sox_create_effects_chain),
      get_symbol(sox_delete_effect),
      get_symbol(sox_delete_effects_chain),
      get_symbol(sox_effect_options),
      get_symbol(sox_find_effect),
      get_symbol(sox_flow_effects),
      get_symbol(sox_get_effect_fns),
      get_symbol(sox_get_format_fns),
      get_symbol(sox_get_globals),
      get_symbol(sox_open_read),
      get_symbol(sox_open_write),
      get_symbol(sox_strerror),
      get_symbol(sox_write)};
#undef get_symbol
}

// Handle to the dlopen-ed libsox
class StubImpl {
  at::DynamicLibrary handle;

 public:
  const LibSoxStub stub;

  StubImpl(const char* name) : handle(name), stub(get_stub(handle)) {
    // Global config
    auto config = stub.sox_get_globals();
    config->verbosity = 0;
    config->use_threads = sox_false;

    // Init sox effect plugins
    auto fn = (int (*)())handle.sym("sox_init");
    TORCH_CHECK(SOX_SUCCESS == fn(), "Failed to initialize sox effects.");
  }

  ~StubImpl() {
    auto fn = (int (*)())handle.sym("sox_quit");
    if (SOX_SUCCESS != fn()) {
      TORCH_WARN("Failed to release sox effect plugins.");
    }
  }
};
} // namespace

#if defined(_WIN32)
#define EXT "lib"
#elif defined(__APPLE__)
#define EXT "dylib"
#else
#define EXT "so"
#endif
const LibSoxStub& libsox_stub() {
  static const StubImpl s{"libsox." EXT};
  return s.stub;
}
#undef EXT

} // namespace torchaudio::sox::detail
