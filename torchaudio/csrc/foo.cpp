#include <dlfcn.h>
#include <torch/script.h>

namespace {

void* handle;

void test_dlopen(const std::string& name) {
  extern void* handle;
  double (*cosine)(double);
  char* error;

  handle = dlopen(name.c_str(), RTLD_LAZY);
  TORCH_CHECK(handle, "Failed to dlopen ", name);

  /*
  int (*sox_init)(void);
  sox_init = (int (*)(void))(dlsym(handle, "sox_init"));
  sox_init();

  cosine = (double (*)(double)) dlsym(handle, "cos");
  if ((error = dlerror()) != NULL)  {
    fputs(error, stderr);
    exit(1);
  }
  std::cerr << "cos(2.) = " << (*cosine)(2.0) << std::endl;
  */
}

void test_dlclose() {
  extern void* handle;
  TORCH_CHECK(handle, "handle is not initialized");
  dlclose(handle);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::testdlopen", &test_dlopen);
  m.def("torchaudio::testdlclose", &test_dlclose);
}

} // namespace
