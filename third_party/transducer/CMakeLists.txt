add_library(warprnnt STATIC submodule/src/rnnt_entrypoint.cpp)
target_compile_definitions(warprnnt PRIVATE RNNT_DISABLE_OMP)
target_include_directories(warprnnt PUBLIC submodule/include)
