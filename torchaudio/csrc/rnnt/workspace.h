#pragma once

#include <cstring>
#include <vector>

#include <torchaudio/csrc/rnnt/options.h>

namespace torchaudio {
namespace rnnt {

// Since CUDA has strict memory alignment, it's better to keep allocated memory
// blocks separate for different data types.

// DtypeWorkspace holds a "view" of  workspace for:
//     1. softmax denominators (in log form), size = B * max_T * max_U
//     2. log probibility pairs for blank and target, size = B * max_T * max_U
//     3. alphas, size = B * max_T * max_U
//     4. betas, size = B * max_T * max_U
template <typename DTYPE>
class DtypeWorkspace {
 public:
  DtypeWorkspace() : options_(), size_(0), data_(nullptr) {}
  DtypeWorkspace(const Options& options, DTYPE* data, int size)
      : DtypeWorkspace() {
    Reset(options, data, size);
  }
  ~DtypeWorkspace() {}

  static int ComputeSizeFromOptions(const Options& options) {
    CHECK_NE(options.device_, UNDEFINED);
    return ComputeSizeForDenominators(options) +
        ComputeSizeForLogProbs(options) + ComputeSizeForAlphas(options) +
        ComputeSizeForBetas(options);
  }

  void Free();
  void Reset(const Options& options, DTYPE* data, int size) {
    int needed_size = ComputeSizeFromOptions(options);
    CHECK_LE(needed_size, size);
    options_ = options;
    data_ = data;
    size_ = size;
  }
  int Size() const {
    return size_;
  }

  DTYPE* GetPointerToDenominators() const {
    return data_;
  }
  DTYPE* GetPointerToLogProbs() const {
    return GetPointerToDenominators() + ComputeSizeForDenominators(options_);
  }
  DTYPE* GetPointerToAlphas() const {
    return GetPointerToLogProbs() + ComputeSizeForLogProbs(options_);
  }
  DTYPE* GetPointerToBetas() const {
    return GetPointerToAlphas() + ComputeSizeForAlphas(options_);
  }

 private:
  static int ComputeSizeForDenominators(const Options& options) { // B * T * U
    return options.BTU();
  }

  static int ComputeSizeForLogProbs(const Options& options) { // B * T * U * 2
    return options.BTU() * 2;
  }

  static int ComputeSizeForAlphas(const Options& options) { // B * T * U
    return options.BTU();
  }

  static int ComputeSizeForBetas(const Options& options) { // B * T * U
    return options.BTU();
  }

  Options options_;
  int size_; // number of elements in allocated memory.
  DTYPE* data_; // pointer to the allocated memory.
};

// IntWorkspace holds a "view" of workspace for:
//     1. alpha counters, size = B * max_U
//     2. beta counters, size = B * max_U
class IntWorkspace {
 public:
  IntWorkspace() : options_(), size_(0), data_(nullptr) {}
  IntWorkspace(const Options& options, int* data, int size) : IntWorkspace() {
    Reset(options, data, size);
  }
  ~IntWorkspace() {}

  static int ComputeSizeFromOptions(const Options& options) {
    return ComputeSizeForAlphaCounters(options) +
        ComputeSizeForBetaCounters(options);
  }

  void Reset(const Options& options, int* data, int size) {
    int needed_size = ComputeSizeFromOptions(options);
    CHECK_LE(needed_size, size);
    options_ = options;
    data_ = data;
    size_ = size;
    ResetAlphaBetaCounters();
  }
  int Size() const {
    return size_;
  }

  int* GetPointerToAlphaCounters() const {
    CHECK_EQ(options_.device_, GPU);
    return data_;
  }
  int* GetPointerToBetaCounters() const {
    CHECK_EQ(options_.device_, GPU);
    return GetPointerToAlphaCounters() + ComputeSizeForAlphaCounters(options_);
  }

 private:
  inline void ResetAlphaBetaCounters() {
#ifdef USE_CUDA
    if (data_ != nullptr && options_.device_ == GPU) {
      cudaMemset(
          GetPointerToAlphaCounters(),
          0,
          ComputeSizeForAlphaCounters(options_) * sizeof(int));
      cudaMemset(
          GetPointerToBetaCounters(),
          0,
          ComputeSizeForBetaCounters(options_) * sizeof(int));
    }
#endif // USE_CUDA
#ifdef USE_ROCM
    if (data_ != nullptr && options_.device_ == GPU) {
      hipMemset(
          GetPointerToAlphaCounters(),
          0,
          ComputeSizeForAlphaCounters(options_) * sizeof(int));
      hipMemset(
          GetPointerToBetaCounters(),
          0,
          ComputeSizeForBetaCounters(options_) * sizeof(int));
    }
#endif // USE_ROCM
  }

  static int ComputeSizeForAlphaCounters(const Options& options) { // B * U
#if defined(USE_CUDA) || defined(USE_ROCM)
    if (options.device_ == GPU) {
      return options.BU();
    } else {
      return 0;
    }
#else
    return 0;
#endif // USE_CUDA
  }
  static int ComputeSizeForBetaCounters(const Options& options) { // B * U
#if defined(USE_CUDA) || defined(USE_ROCM)
    if (options.device_ == GPU) {
      return options.BU();
    } else {
      return 0;
    }
#else
    return 0;
#endif // USE_CUDA
  }

  Options options_;
  int size_; // number of elements in allocated memory.
  int* data_; // pointer to the allocated memory.
};

// Workspace<DTYPE> holds:
//     1. DtypeWorkspace<DTYPE>
//     2. IntWorkspace
template <typename DTYPE>
class Workspace {
 public:
  Workspace() : options_(), dtype_workspace_(), int_workspace_() {}
  Workspace(
      const Options& options,
      DTYPE* dtype_data,
      int dtype_size,
      int* int_data,
      int int_size)
      : Workspace() {
    Reset(options, dtype_data, dtype_size, int_data, int_size);
  }
  ~Workspace() {}

  void Reset(
      const Options& options,
      DTYPE* dtype_data,
      int dtype_size,
      int* int_data,
      int int_size) {
    options_ = options;
    dtype_workspace_.Reset(options_, dtype_data, dtype_size);
    int_workspace_.Reset(options_, int_data, int_size);
  }

  const Options& GetOptions() const {
    return options_;
  }

  DTYPE* GetPointerToDenominators() const {
    return dtype_workspace_.GetPointerToDenominators();
  }
  DTYPE* GetPointerToLogProbs() const {
    return dtype_workspace_.GetPointerToLogProbs();
  }
  DTYPE* GetPointerToAlphas() const {
    return dtype_workspace_.GetPointerToAlphas();
  }
  DTYPE* GetPointerToBetas() const {
    return dtype_workspace_.GetPointerToBetas();
  }
  int* GetPointerToAlphaCounters() const {
    return int_workspace_.GetPointerToAlphaCounters();
  }
  int* GetPointerToBetaCounters() const {
    return int_workspace_.GetPointerToBetaCounters();
  }

 private:
  Options options_;
  DtypeWorkspace<DTYPE> dtype_workspace_;
  IntWorkspace int_workspace_;
};

} // namespace rnnt
} // namespace torchaudio
