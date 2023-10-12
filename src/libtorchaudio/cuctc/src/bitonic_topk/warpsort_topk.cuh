/**
 *  Modified from Rapidsai/raft(https://github.com/rapidsai/raft)
 *
 */

#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>
#include "bitonic_sort.cuh"
#include "pow2_utils.cuh"

namespace cu_ctc {

/*
  Three APIs of different scopes are provided:
    1. host function: warp_sort_topk()
    2. block-wide API: class block_sort
    3. warp-wide API: class warp_sort_filtered and class warp_sort_immediate


  1. warp_sort_topk()
    (see the docstring)

  2. class block_sort
    It can be regarded as a fixed size priority queue for a thread block,
    although the API is not typical.
    class warp_sort_filtered and warp_sort_immediate can be used to instantiate
  block_sort.

    It uses dynamic shared memory as an intermediate buffer.
    So the required shared memory size should be calculated using
    calc_smem_size_for_block_wide() and passed as the 3rd kernel launch
  parameter.

    To add elements to the queue, use add(T val, IdxT idx) with unique values
  per-thread. Use WarpSortClass<...>::kDummy constant for the threads outside of
  input bounds.

    After adding is finished, function done() should be called. And finally,
  store() is used to get the top-k result.

    Example:
      __global__ void kernel() {
        block_sort<warp_sort_immediate, ...> queue(...);

        for (IdxT i = threadIdx.x; i < len, i += blockDim.x) {
          queue.add(in[i], in_idx[i]);
        }

        queue.done();
        queue.store(out, out_idx);
     }

     int smem_size = calc_smem_size_for_block_wide<T>(...);
     kernel<<<grid_dim, block_dim, smem_size>>>();


  3. class warp_sort_filtered and class warp_sort_immediate
    These two classes can be regarded as fixed size priority queue for a warp.
    Usage is similar to class block_sort. No shared memory is needed.

    The host function (warp_sort_topk) uses a heuristic to choose between these
  two classes for sorting, warp_sort_immediate being chosen when the number of
  inputs per warp is somewhat small (see the usage of
  LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing).

    Example:
      __global__ void kernel() {
        warp_sort_immediate<...> queue(...);
        int warp_id = threadIdx.x / WarpSize;
        int lane_id = threadIdx.x % WarpSize;

        for (IdxT i = lane_id; i < len, i += WarpSize) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        // each warp outputs to a different offset
        queue.store(out + warp_id * k, out_idx + warp_id * k);
      }
 */

namespace topk {
static constexpr int kMaxCapacity = 256;
/** Whether 'left` should indeed be on the left w.r.t. `right`. */
template <bool Ascending, typename T>
__device__ __forceinline__ auto is_ordered(T left, T right) -> bool {
  if constexpr (Ascending) {
    return left < right;
  }
  if constexpr (!Ascending) {
    return left > right;
  }
}

constexpr inline auto calc_capacity(int k) -> int {
  int capacity = isPo2(k) ? k : (1 << (log2(k) + 1));
  return capacity;
}

/**
 * A fixed-size warp-level priority queue.
 * By feeding the data through this queue, you get the `k <= Capacity`
 * smallest/greatest values in the data.
 *
 * @tparam Capacity
 *   maximum number of elements in the queue.
 * @tparam Ascending
 *   which comparison to use: `true` means `<`, collect the smallest elements,
 *   `false` means `>`, collect the greatest elements.
 * @tparam T
 *   the type of keys (what is being compared)
 * @tparam IdxT
 *   the type of payload (normally, indices of elements), i.e.
 *   the content sorted alongside the keys.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort {
  static_assert(isPo2(Capacity));

 public:
  /**
   *  The `empty` value for the chosen binary operation,
   *  i.e. `Ascending ? upper_bound<T>() : lower_bound<T>()`.
   */
  static constexpr T kDummy = Ascending ? upper_bound<T>() : lower_bound<T>();
  /** Width of the subwarp. */
  static constexpr int kWarpWidth = std::min<int>(Capacity, WarpSize);
  /** The number of elements to select. */
  const int k;

  /**
   * Construct the warp_sort empty queue.
   *
   * @param k
   *   number of elements to select.
   */
  __device__ warp_sort(int k) : k(k) {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_arr_[i] = kDummy;
    }
  }

  /**
   * Load k values from the pointers at the given position, and merge them in
   * the storage.
   *
   * When it actually loads the values, it always performs some collective warp
   * operations in the end, thus enforcing warp sync. This means, it's safe to
   * call `store` with the same arguments after `load_sorted` without extra
   * sync. Note, however, that this is not neccesarily true for the reverse
   * order, because the access patterns of `store` and `load_sorted` are
   * different.
   *
   * @param[in] in
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] in_idx
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] do_merge
   *    must be the same for all threads within a subwarp of size `kWarpWidth`.
   *    It serves as a conditional; when `false` the function does nothing.
   *    We need it to ensure threads within a full warp don't diverge calling
   * `bitonic::merge()`.
   */
  __device__ void load_sorted(
      const T* in,
      const IdxT* in_idx,
      bool do_merge = true) {
    if (do_merge) {
      int idx = Pow2<kWarpWidth>::mod(laneId()) ^ Pow2<kWarpWidth>::Mask;
#pragma unroll
      for (int i = kMaxArrLen - 1; i >= 0; --i, idx += kWarpWidth) {
        if (idx < k) {
          T t = in[idx];
          if (is_ordered<Ascending>(t, val_arr_[i])) {
            val_arr_[i] = t;
            idx_arr_[i] = in_idx[idx];
          }
        }
      }
    }
    if (kWarpWidth < WarpSize || do_merge) {
      topk::bitonic<kMaxArrLen>(Ascending, kWarpWidth)
          .merge(val_arr_, idx_arr_);
    }
  }

  /**
   *  Save the content by the pointer location.
   *
   * @param[out] out
   *   device pointer to a contiguous array, unique per-subwarp of size
   * `kWarpWidth` (length: k <= kWarpWidth * kMaxArrLen).
   * @param[out] out_idx
   *   device pointer to a contiguous array, unique per-subwarp of size
   * `kWarpWidth` (length: k <= kWarpWidth * kMaxArrLen).
   */
  __device__ void store(T* out, IdxT* out_idx) const {
    int idx = Pow2<kWarpWidth>::mod(laneId());
#pragma unroll kMaxArrLen
    for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
      out[idx] = val_arr_[i];
      out_idx[idx] = idx_arr_[i];
    }
  }

 protected:
  static constexpr int kMaxArrLen = Capacity / kWarpWidth;

  T val_arr_[kMaxArrLen];
  IdxT idx_arr_[kMaxArrLen];

  /**
   * Merge another array (sorted in the opposite direction) in the queue.
   * Thanks to the other array being sorted in the opposite direction,
   * it's enough to call bitonic.merge once to maintain the valid state
   * of the queue.
   *
   * @tparam PerThreadSizeIn
   *   the size of the other array per-thread (compared to `kMaxArrLen`).
   *
   * @param keys_in
   *   the values to be merged in. Pointers are unique per-thread. The values
   *   must already be sorted in the opposite direction.
   *   The layout of `keys_in` must be the same as the layout of `val_arr_`.
   * @param ids_in
   *   the associated indices of the elements in the same format as `keys_in`.
   */
  template <int PerThreadSizeIn>
  __device__ __forceinline__ void merge_in(
      const T* __restrict__ keys_in,
      const IdxT* __restrict__ ids_in) {
#pragma unroll
    for (int i = std::min(kMaxArrLen, PerThreadSizeIn); i > 0; i--) {
      T& key = val_arr_[kMaxArrLen - i];
      T other = keys_in[PerThreadSizeIn - i];
      if (is_ordered<Ascending>(other, key)) {
        key = other;
        idx_arr_[kMaxArrLen - i] = ids_in[PerThreadSizeIn - i];
      }
    }
    topk::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
  }
};

/**
 * This version of warp_sort compares each input element against the current
 * estimate of k-th value before adding it to the intermediate sorting buffer.
 * This makes the algorithm do less sorting steps for long input sequences
 * at the cost of extra checks on each step.
 *
 * This implementation is preferred for large len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_filtered : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;

  __device__ warp_sort_filtered(int k)
      : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0), k_th_(kDummy) {
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
    }
  }

  __device__ void add(T val, IdxT idx) {
    // comparing for k_th should reduce the total amount of updates:
    // `false` means the input value is surely not in the top-k values.
    bool do_add = is_ordered<Ascending>(val, k_th_);
    // merge the buf if it's full and we cannot add an element anymore.
    if (any(buf_len_ + do_add > kMaxBufLen)) {
      // still, add an element before merging if possible for this thread
      if (do_add && buf_len_ < kMaxBufLen) {
        add_to_buf_(val, idx);
        do_add = false;
      }
      merge_buf_();
    }
    // add an element if necessary and haven't already.
    if (do_add) {
      add_to_buf_(val, idx);
    }
  }

  __device__ void done() {
    if (any(buf_len_ != 0)) {
      merge_buf_();
    }
  }

 private:
  __device__ __forceinline__ void set_k_th_() {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    // const int id = (k - 1) / kWarpWidth;
    const int id = Pow2<kWarpWidth>::div(k - 1);
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      if (i == id) {
        k_th_ = shfl(val_arr_[i], k - 1, kWarpWidth);
      }
    }
    // k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  __device__ __forceinline__ void merge_buf_() {
    topk::bitonic<kMaxBufLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
    this->merge_in<kMaxBufLen>(val_buf_, idx_buf_);
    buf_len_ = 0;
    set_k_th_(); // contains warp sync
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
    }
  }

  __device__ __forceinline__ void add_to_buf_(T val, IdxT idx) {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }
    buf_len_++;
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  static constexpr int kMaxBufLen = (Capacity <= 64) ? 2 : 4;

  T val_buf_[kMaxBufLen];
  IdxT idx_buf_[kMaxBufLen];
  int buf_len_;

  T k_th_;
};

/**
 * This version of warp_sort adds every input element into the intermediate
 * sorting buffer, and thus does the sorting step every `Capacity` input
 * elements.
 *
 * This implementation is preferred for very small len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_immediate : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;

  __device__ warp_sort_immediate(int k)
      : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0) {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_buf_[i] = kDummy;
    }
  }

  __device__ void add(T val, IdxT idx) {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxArrLen; ++i) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }

    ++buf_len_;
    if (buf_len_ == kMaxArrLen) {
      topk::bitonic<kMaxArrLen>(!Ascending, kWarpWidth)
          .sort(val_buf_, idx_buf_);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
#pragma unroll
      for (int i = 0; i < kMaxArrLen; i++) {
        val_buf_[i] = kDummy;
      }
      buf_len_ = 0;
    }
  }

  __device__ void done() {
    if (buf_len_ != 0) {
      topk::bitonic<kMaxArrLen>(!Ascending, kWarpWidth)
          .sort(val_buf_, idx_buf_);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
    }
  }

 private:
  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  T val_buf_[kMaxArrLen];
  IdxT idx_buf_[kMaxArrLen];
  int buf_len_;
};

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr inline __host__ __device__ IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}
template <typename IntType>
constexpr inline __device__ IntType roundUp256(IntType num) {
  // return (num + 255) / 256 * 256;
  constexpr int MASK = 255;
  return (num + MASK) & (~MASK);
}

template <typename T, typename IdxT>
auto calc_smem_size_for_block_wide(int num_of_subwarp, int k) -> int {
  return roundUp256(ceildiv(num_of_subwarp, 2) * sizeof(T) * k) +
      ceildiv(num_of_subwarp, 2) * sizeof(IdxT) * k;
}

template <
    template <int, bool, typename, typename>
    class WarpSortWarpWide,
    int Capacity,
    bool Ascending,
    typename T,
    typename IdxT>
class block_sort {
  using queue_t = WarpSortWarpWide<Capacity, Ascending, T, IdxT>;

 public:
  __device__ block_sort(int k, uint8_t* smem_buf) : queue_(k) {
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    const int num_of_warp = subwarp_align::div(blockDim.x);
    idx_smem_ = reinterpret_cast<IdxT*>(
        smem_buf + roundUp256(ceildiv(num_of_warp, 2) * sizeof(T) * k));
  }

  __device__ void add(T val, IdxT idx) {
    queue_.add(val, idx);
  }

  /**
   * At the point of calling this function, the warp-level queues consumed all
   * input independently. The remaining work to be done is to merge them
   * together.
   *
   * Here we tree-merge the results using the shared memory and block sync.
   */
  __device__ void done() {
    queue_.done();

    const int warp_id = subwarp_align::div(threadIdx.x);
    // NB: there is no need for the second __synchthreads between .load_sorted
    // and .store:
    //     we shift the pointers every iteration, such that individual warps
    //     either access the same locations or do not overlap with any of the
    //     other warps. The access patterns within warps are different for the
    //     two functions, but .load_sorted implies warp sync at the end, so
    //     there is no need for __syncwarp either.
    for (int shift_mask = ~0,
             nwarps = subwarp_align::div(blockDim.x),
             split = (nwarps + 1) >> 1;
         nwarps > 1;
         nwarps = split, split = (nwarps + 1) >> 1) {
      if (warp_id < nwarps && warp_id >= split) {
        int dst_warp_shift = (warp_id - (split & shift_mask)) * queue_.k;
        queue_.store(val_smem_ + dst_warp_shift, idx_smem_ + dst_warp_shift);
      }
      __syncthreads();

      shift_mask = ~shift_mask; // invert the mask
      {
        int src_warp_shift = (warp_id + (split & shift_mask)) * queue_.k;
        // The last argument serves as a condition for loading
        //  -- to make sure threads within a full warp do not diverge on
        //  `bitonic::merge()`
        queue_.load_sorted(
            val_smem_ + src_warp_shift,
            idx_smem_ + src_warp_shift,
            warp_id < nwarps - split);
      }
    }
  }

  /** Save the content by the pointer location. */
  __device__ void store(T* out, IdxT* out_idx) const {
    if (threadIdx.x < subwarp_align::Value) {
      queue_.store(out, out_idx);
    }
  }

 private:
  using subwarp_align = Pow2<queue_t::kWarpWidth>;
  queue_t queue_;
  T* val_smem_;
  IdxT* idx_smem_;
};

} // namespace topk

} // namespace cu_ctc
