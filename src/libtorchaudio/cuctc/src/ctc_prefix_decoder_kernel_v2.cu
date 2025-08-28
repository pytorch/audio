// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <float.h>
#include <algorithm>
#include <limits>
#include "../include/ctc_prefix_decoder_host.h"
#include "ctc_fast_divmod.cuh"
#include "cub/cub.cuh"
#include "device_data_wrap.h"
#include "device_log_prob.cuh"

#include "bitonic_topk/warpsort_topk.cuh"

namespace cu_ctc {
__inline__ __device__ float _lauguage() {
  return 1.0f;
}
__inline__ __device__ float _logprob(float a, float b) {
  return a + b;
}
__inline__ __device__ float _logsumexp(float a, float b) {
  float max_ab = a > b ? a : b;
  float neg_abs_ab = (a - b) > 0 ? (b - a) : (a - b);
  return max_ab + __logf(1 + __expf(neg_abs_ab));
}
__inline__ __device__ bool compare(int len, int* a, int* b) {
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return 0;
  return 1;
}

template <
    int BLOCK_SIZE,
    int ITEMS_PER_THREAD,
    typename KeyT,
    typename ValueT,
    typename BLOCK_TOPK_FUN,
    typename SET_KEY_VALUE_FUN>
__device__ __forceinline__ void block_topk_striped_wrap_with_default_key(
    KeyT (&keys)[ITEMS_PER_THREAD],
    ValueT (&values)[ITEMS_PER_THREAD],
    const int k,
    const int valid_count_this_block,
    const KeyT default_key,
    BLOCK_TOPK_FUN& block_topk_fun,
    SET_KEY_VALUE_FUN& set_key_value_fun) {
  const int tx = threadIdx.x;
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    int idx = BLOCK_SIZE * ITEM + tx;
    if (idx < valid_count_this_block) {
      set_key_value_fun(keys[ITEM], values[ITEM], idx);
    } else {
      keys[ITEM] = default_key;
    }
  }
  const int valid_count_this_iter =
      (valid_count_this_block < (BLOCK_SIZE * ITEMS_PER_THREAD))
      ? valid_count_this_block
      : (BLOCK_SIZE * ITEMS_PER_THREAD);
  block_topk_fun(keys, values, k, valid_count_this_iter);
  __syncthreads();
  const int stride = BLOCK_SIZE * ITEMS_PER_THREAD - k;
  for (int idx_offset = ITEMS_PER_THREAD * BLOCK_SIZE;
       idx_offset < valid_count_this_block;
       idx_offset += stride) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      int local_idx = BLOCK_SIZE * ITEM + tx - k;
      int target_idx = idx_offset + local_idx;
      if (local_idx >= 0 && target_idx < valid_count_this_block) {
        set_key_value_fun(keys[ITEM], values[ITEM], target_idx);
      }
      if (target_idx >= valid_count_this_block) {
        keys[ITEM] = default_key;
      }
    }
    const int iter_valid_count =
        ((valid_count_this_block - idx_offset) >= stride)
        ? (BLOCK_SIZE * ITEMS_PER_THREAD)
        : (k + valid_count_this_block - idx_offset);
    block_topk_fun(keys, values, k, iter_valid_count);
    __syncthreads();
  }
}

__global__ void prob_matrix_v2_kernel(
    LogProb log_prob_struct,
    int step,
    float2* pprev,
    float* ptable,
    float* ptablen,
    int* clast,
    int lc,
    int ldc,
    int beam,
    int ldbeam,
    int bs,
    int blid,
    int spid)

{
  const int batch_id = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  if (!log_prob_struct.need_process_on_ith_step(batch_id, step))
    return;
  const int select_seq =
      log_prob_struct.ith_selected_seq_in_this_batch(batch_id, step);
  if (batch_id >= bs || tid >= (lc * beam))
    return;
  for (; tid < (lc * beam); tid += stride) {
    int beamid = tid / lc;
    int charid = tid - beamid * lc;

    if ((charid != blid) && charid != spid) {
      int idout = charid + (beamid + batch_id * beam) * ldc;
      int target_clast = clast[batch_id * ldbeam + beamid];

      float cur_prob = log_prob_struct.at(batch_id, select_seq, charid);
      float out_prob;
      float2 beamid_p = pprev[batch_id * ldbeam + beamid];
      if (target_clast == charid) {
        out_prob = _logprob(cur_prob, beamid_p.x);

        float out_prob_prefix = _logprob(cur_prob, beamid_p.y);
        int idout_prefix = blid + (batch_id * beam + beamid) * ldc;
        ptablen[idout_prefix] = out_prob_prefix;
      } else {
        out_prob = _logprob(cur_prob, _logsumexp(beamid_p.x, beamid_p.y));
      }
      ptable[idout] = -FLT_MAX;
      ptablen[idout] = out_prob;
    }
  }
}

__global__ void prob_space_blank_kernel_v2(
    LogProb log_prob_struct,
    int step,
    float2* pprev,
    float* ptable,
    float* ptablen,
    int* clast,
    int lc,
    int ldc,
    int beam,
    int ldbeam,
    int bs,
    int blid,
    int spid) {
  const int batch_id = blockIdx.y;
  if (!log_prob_struct.need_process_on_ith_step(batch_id, step))
    return;
  const int select_seq =
      log_prob_struct.ith_selected_seq_in_this_batch(batch_id, step);
  const int beamid = threadIdx.x;
  if (beamid < beam) {
    // assume blank at 0
    float pc = log_prob_struct.at(batch_id, select_seq, blid);
    float2 tmpprev = pprev[batch_id * ldbeam + beamid];
    int last_char = clast[batch_id * ldbeam + beamid];
    int idout = blid + (batch_id * beam + beamid) * ldc;
    ptable[idout] = _logprob(pc, _logsumexp(tmpprev.x, tmpprev.y));
    if (last_char == blid)
      ptablen[idout] = -FLT_MAX;
  }

  if (spid >= 0 && (spid != blid) && beamid < beam) {
    float pc = log_prob_struct.at(batch_id, select_seq, spid);
    float2 tmpprev = pprev[batch_id * ldbeam + beamid];
    int idout = spid + (batch_id * beam + beamid) * ldc;
    ptablen[idout] = _lauguage() *
        _logprob(pc, _logsumexp(tmpprev.x, tmpprev.y)); // logsumexp
    ptable[idout] = -FLT_MAX;
  }
}

__global__ void matrix_merge_kernel_v2(
    LogProb log_prob_struct,
    int step,
    float* ptable,
    float* ptablen,
    int* ptid,
    int* clast,
    int* clist,
    int* clen,
    int lc,
    int ldc,
    int beam,
    int ldbeam,
    int ldseq_len,
    int bs,
    int blid) {
  // not produce the "l+ not in Aprev" part. If do this, need use ptalbe(n)@t-1
  // this is a little kernel & latency dependency & almost no parallel
  // each thread produce one beam .vs. one beam.
  // block=beam,thread=beam (if beam<32, can use one block for optim)

  const int batch_id = blockIdx.y;
  if (!log_prob_struct.need_process_on_ith_step(batch_id, step))
    return;
  const int select_seq =
      log_prob_struct.ith_selected_seq_in_this_batch(batch_id, step);
  __shared__ int tmpclen[128]; // beam<128
  int tidin, tidout;

  if (threadIdx.x < beam) {
    tmpclen[threadIdx.x] = clen[threadIdx.x + blockIdx.y * ldbeam];
  }
  __syncthreads();
  if (threadIdx.x < beam &&
      ((tmpclen[threadIdx.x] - 1) ==
       tmpclen[blockIdx.x])) { // char=blank && belong to the same beam @t-1; if
                               // not meet, the whole block will not calculate.
                               // delta(L)=1
    if (compare(
            tmpclen[blockIdx.x],
            clist + threadIdx.x * ldseq_len + blockIdx.y * ldseq_len * beam,
            clist + blockIdx.x * ldseq_len + blockIdx.y * ldseq_len * beam)) {
      tidin = clast[threadIdx.x + blockIdx.y * ldbeam] +
          (blockIdx.x + blockIdx.y * beam) * ldc;
      tidout = blid + (threadIdx.x + blockIdx.y * beam) * ldc;

      ptable[tidout] = _logsumexp(ptable[tidout], ptable[tidin]);
      ptablen[tidout] = _logsumexp(ptablen[tidout], ptablen[tidin]);
      ptable[tidin] = -FLT_MAX;
      ptablen[tidin] = -FLT_MAX;
    }
  }
}

template <int BLOCK_SIZE, int Capacity>
__global__ __launch_bounds__(BLOCK_SIZE) void first_matrix__bitonic_topk_kernel(
    LogProb log_prob_struct,
    int step,
    float2* pprev,
    int* ptid,
    int* clast,
    int* clen,
    int* clist,
    int beam,
    int ldbeam,
    int ldseq_len,
    int blid,
    int bs,
    float* score,
    int smem_result_byte_offset) {
  const int batch_id = blockIdx.x;
  const int tx = threadIdx.x;
  if (!log_prob_struct.need_process_on_ith_step(batch_id, step))
    return;
  const bool is_need_add_blank = log_prob_struct.need_add_blank(batch_id, step);
  const int select_seq =
      log_prob_struct.ith_selected_seq_in_this_batch(batch_id, step);
  const int vocab_size = log_prob_struct.vocab_size;

  extern __shared__ __align__(256) uint8_t smem_buf_bytes[];
  constexpr bool Ascending = false;
  using namespace cu_ctc::topk;
  block_sort<warp_sort_filtered, Capacity, Ascending, float, int> queue(
      beam, smem_buf_bytes);
  const int per_thread_lim = vocab_size + laneId();

  for (int id = tx; id < per_thread_lim; id += BLOCK_SIZE) {
    float key = (id < vocab_size)
        ? (log_prob_struct.at(batch_id, select_seq, id))
        : (warp_sort_filtered<Capacity, Ascending, float, int>::kDummy);
    int value = id;
    queue.add(key, value);
  }
  queue.done();
  float* block_topk_key =
      reinterpret_cast<float*>(smem_buf_bytes + smem_result_byte_offset);
  int* block_topk_value =
      reinterpret_cast<int*>(block_topk_key + sizeof(float) * beam);

  queue.store(block_topk_key, block_topk_value);
  for (int idx = tx; idx < beam; idx += BLOCK_SIZE) {
    int id = block_topk_value[idx];
    float key = block_topk_key[idx];
    int shift = clen[idx + batch_id * ldbeam];
    if (id != blid) {
      float2 xy =
          is_need_add_blank ? float2{key, -FLT_MAX} : float2{-FLT_MAX, key};
      pprev[batch_id * ldbeam + idx] = xy;
      clist[batch_id * beam * ldseq_len + idx * ldseq_len + shift] = id;
      clen[batch_id * ldbeam + idx] += 1;
      clast[batch_id * ldbeam + idx] = id;
    } else {
      pprev[batch_id * ldbeam + idx] = float2{key, -FLT_MAX};
    }
    score[batch_id * ldbeam + idx] = key;
  }
}

template <int BLOCK_SIZE, int Capacity>
__global__
__launch_bounds__(BLOCK_SIZE) void bitonic_topk_multi_block_per_batch_kernel(
    LogProb log_prob_struct,
    int step,
    const float* ptable,
    const float* ptablen,
    int lc,
    int ldc,
    int beam,
    int bs,
    float* topk_key_buffer,
    int* topk_value_buffer,
    FastDivmod ldc_fast_divmod) {
  const int batch_id = blockIdx.y;
  if (batch_id >= bs)
    return;
  if (!log_prob_struct.need_process_on_ith_step(batch_id, step))
    return;
  extern __shared__ __align__(256) uint8_t smem_buf_bytes[];
  constexpr bool Ascending = false;
  using namespace cu_ctc::topk;
  block_sort<warp_sort_filtered, Capacity, Ascending, float, int> queue(
      beam, smem_buf_bytes);
  const int bx = blockIdx.x;
  const int blocks_per_batch = gridDim.x;
  const int all_items_per_batch = ldc * beam;
  const int stride = blocks_per_batch * BLOCK_SIZE;
  const int gid = threadIdx.x + bx * BLOCK_SIZE;
  const int block_out_offset = (batch_id * blocks_per_batch + bx) * beam;
  const int per_thread_lim = all_items_per_batch + laneId();

  for (int id = gid; id < per_thread_lim; id += stride) {
    float key = warp_sort_filtered<Capacity, Ascending, float, int>::kDummy;
    int value = id;
    if (id < all_items_per_batch) {
      int quotient;
      int reminder;
      ldc_fast_divmod(quotient, reminder, id); // reminder = id%lc;
      if (reminder < lc) {
        int tidin = batch_id * all_items_per_batch + id;
        float p = ptable[tidin];
        float pn = ptablen[tidin];
        key = _logsumexp(p, pn);
      }
    }
    queue.add(key, value);
  }
  queue.done();

  queue.store(
      topk_key_buffer + block_out_offset, topk_value_buffer + block_out_offset);
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD, int WRITE_THREDS = 8>
__global__
__launch_bounds__(BLOCK_SIZE) void topk_reduce_and_copy_list_per_batch_kernel(
    LogProb log_prob_struct,
    int step,
    int beam,
    int items_per_batch,
    int bs,
    float* topk_key_buffer,
    int* topk_value_buffer,
    int ldc,
    int ldbeam,
    int ldseq_len,
    float2* pprev,
    float* ptable,
    float* ptablen,
    int* clast,
    int* clen,
    int* clen2,
    int* clist,
    int* clist2,
    int blid,
    float* score) {
  constexpr int MAX_SUPPORT_BEAM = 128;
  int batch_id = blockIdx.x;
  int rw_offset_this_block = batch_id * items_per_batch;
  if (batch_id >= bs)
    return;
  if (!log_prob_struct.need_process_on_ith_step(batch_id, step))
    return;
  const bool is_need_add_blank = log_prob_struct.need_add_blank(batch_id, step);
  const int tx = threadIdx.x;

  using BlockRadixSortT =
      cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, int>;

  __shared__ union {
    typename BlockRadixSortT::TempStorage temp_storage;
#ifdef USE_PARALLEL_WRITE
    constexpr int smem_size = MAX_SUPPORT_BEAM * (sizeof(float) + sizeof(int));
    uint8_t topk_key_value_smem[smem_size];
#endif
    /* data */
  } ShareSmem;

  float topk_keys[ITEMS_PER_THREAD];
  int topk_values[ITEMS_PER_THREAD];

  auto block_topk_fun = [&](float(&keys)[ITEMS_PER_THREAD],
                            int(&values)[ITEMS_PER_THREAD],
                            const int k,
                            const int valid_count_this_iter) {
    BlockRadixSortT{ShareSmem.temp_storage}.SortDescendingBlockedToStriped(
        keys, values);
  };
  auto set_key_value = [&](float& key, int& value, int idx) {
    key = topk_key_buffer[idx + rw_offset_this_block];
    value = topk_value_buffer[idx + rw_offset_this_block];
  };

  block_topk_striped_wrap_with_default_key<
      BLOCK_SIZE,
      ITEMS_PER_THREAD,
      float,
      int>(
      topk_keys,
      topk_values,
      beam,
      items_per_batch,
#if CUDA_VERSION >= 12090  // CUDA 12.9 and later
      std::numeric_limits<float>::lowest(),
#else
      cub::FpLimits<float>::Lowest(),
#endif
      block_topk_fun,
      set_key_value);

  // write result in global memory
  __syncthreads();

#ifdef USE_PARALLEL_WRITE

  float* smem_keys =
      reinterpret_cast<float*>(&(ShareSmem.topk_key_value_smem[0]));
  int* smem_values = reinterpret_cast<int*>(
      ShareSmem.topk_key_value_smem + MAX_SUPPORT_BEAM * sizeof(float));
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    int idx = BLOCK_SIZE * ITEM + tx;
    if (idx < beam) {
      smem_keys[idx] = topk_keys[ITEM];
      smem_values[idx] = topk_values[ITEM];
    }
  }

  __syncthreads();

  const int sub_warp_id = tx / WRITE_THREDS;
  const int tid_in_subw = tx % WRITE_THREDS;
  const int sub_warps = BLOCK_SIZE / WRITE_THREDS;
  for (int out_beamid = sub_warp_id; out_beamid < beam;
       out_beamid += sub_warps) {
    int id = smem_values[out_beamid];
    int beamid = id / ldc;
    int charid = id - beamid * ldc; // id%ldc
    int prevlen = clen[beamid + batch_id * ldbeam];
    // PARALLEL_WRITE
    for (int i = tid_in_subw; i < prevlen; i += WRITE_THREDS) {
      clist2[batch_id * beam * ldseq_len + out_beamid * ldseq_len + i] =
          clist[batch_id * beam * ldseq_len + beamid * ldseq_len + i];
    }
    if (tid_in_subw == 0) {
      if (charid == blid) {
        clast[batch_id * ldbeam + out_beamid] =
            clast[beamid + batch_id * ldbeam];
        clen2[batch_id * ldbeam + out_beamid] = prevlen;
      } else {
        clast[batch_id * ldbeam + out_beamid] = charid;
        clen2[batch_id * ldbeam + out_beamid] = prevlen + 1;
        clist2[batch_id * beam * ldseq_len + out_beamid * ldseq_len + prevlen] =
            charid;
      }
      float2 ptable_ptablen;
      ptable_ptablen.x = ptable[batch_id * ldc * beam + id];
      ptable_ptablen.y = ptablen[batch_id * ldc * beam + id];
      float cur_score = _logsumexp(ptable_ptablen.x, ptable_ptablen.y);
      score[batch_id * ldbeam + out_beamid] = cur_score;
      float2 ptable_ptablen2 = float2{cur_score, -FLT_MAX};
      pprev[batch_id * ldbeam + out_beamid] =
          is_need_add_blank ? ptable_ptablen2 : ptable_ptablen;
    }
  }
#else
  {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
      int idx = BLOCK_SIZE * ITEM + tx;
      if (idx < beam) {
        int id = topk_values[ITEM];
        int beamid = id / ldc;
        int charid = id - beamid * ldc; // id%ldc
        int prevlen = clen[beamid + batch_id * ldbeam];
        int prevclast = clast[beamid + batch_id * ldbeam];
        for (int i = 0; i < prevlen; i++) {
          clist2[batch_id * beam * ldseq_len + idx * ldseq_len + i] =
              clist[batch_id * beam * ldseq_len + beamid * ldseq_len + i];
        }
        if (charid == blid) {
          clast[batch_id * ldbeam + idx] = prevclast;
          clen2[batch_id * ldbeam + idx] = prevlen;
        } else {
          clast[batch_id * ldbeam + idx] = charid;
          clen2[batch_id * ldbeam + idx] = prevlen + 1;
          clist2[batch_id * beam * ldseq_len + idx * ldseq_len + prevlen] =
              charid;
        }

        float2 ptable_ptablen;
        ptable_ptablen.x = ptable[batch_id * ldc * beam + id];
        ptable_ptablen.y = ptablen[batch_id * ldc * beam + id];
        float cur_score = _logsumexp(ptable_ptablen.x, ptable_ptablen.y);
        score[batch_id * ldbeam + idx] = cur_score;
        float2 ptable_ptablen2 = float2{cur_score, -FLT_MAX};
        pprev[batch_id * ldbeam + idx] =
            is_need_add_blank ? ptable_ptablen2 : ptable_ptablen;
      }
    }
  }
#endif
}

int CTC_prob_matrix_V2(
    LogProb* log_prob_struct,
    int step,
    float2* pprev,
    float* ptable,
    float* ptablen,
    int* clast,
    int lc,
    int ldc,
    int beam,
    int ldbeam,
    int bs,
    int blid,
    int spid,
    cudaStream_t stream) {
  dim3 grid, block;
  block.x = 256, block.y = 1, block.z = 1;
  grid.x = min((lc * beam + block.x - 1) / block.x, MAX_BLOCKS / bs);
  grid.y = bs;
  grid.z = 1;
  prob_matrix_v2_kernel<<<grid, block, 0, stream>>>(
      (*log_prob_struct),
      step,
      pprev,
      ptable,
      ptablen,
      clast,
      lc,
      ldc,
      beam,
      ldbeam,
      bs,
      blid,
      spid);
  block.x = ldbeam, block.y = 1, block.z = 1;
  grid.x = 1, grid.y = bs, grid.z = 1;
  CHECK(ldbeam <= 1024, " only support  beam<=1024");
  prob_space_blank_kernel_v2<<<grid, block, 0, stream>>>(
      (*log_prob_struct),
      step,
      pprev,
      ptable,
      ptablen,
      clast,
      lc,
      ldc,
      beam,
      ldbeam,
      bs,
      blid,
      spid);
  return 0;
}

int CTC_prob_first_step_V2(
    LogProb* log_prob_struct,
    int step,
    float2* pprev,
    int* ptid,
    int* clast,
    int* clen,
    int* clist,
    int beam,
    int ldbeam,
    int ldseq_len,
    int bs,
    float* score,
    cudaStream_t stream,
    int blid) {
  CHECK(beam <= 128, "ERROR: only support beam size <=128 ");

  constexpr int threads_per_block = 256;
  const int grid = bs;

  constexpr int Capacity = 16;
  using FunType =
      decltype(first_matrix__bitonic_topk_kernel<threads_per_block, Capacity>);
  static FunType* FirstMatrixFuns[5]{
      first_matrix__bitonic_topk_kernel<threads_per_block, 8>,
      first_matrix__bitonic_topk_kernel<threads_per_block, 16>,
      first_matrix__bitonic_topk_kernel<threads_per_block, 32>,
      first_matrix__bitonic_topk_kernel<threads_per_block, 64>,
      first_matrix__bitonic_topk_kernel<threads_per_block, 128>};
  int need_capacity = topk::calc_capacity(beam);
  int fun_idx = 0;
  fun_idx = std::max(0, 31 - clz(need_capacity) - 3);
  int actual_capacity = (1 << (fun_idx + 3));
  int num_of_subwarp = threads_per_block / std::min<int>(32, actual_capacity);
  int block_sort_smem_size = cu_ctc::topk::roundUp256(
      cu_ctc::topk::calc_smem_size_for_block_wide<float, int>(
          num_of_subwarp, beam));
  int smem_size =
      block_sort_smem_size + beam * sizeof(float) + beam * sizeof(int);
  auto kernel = FirstMatrixFuns[fun_idx];
  kernel<<<grid, threads_per_block, smem_size, stream>>>(
      (*log_prob_struct),
      step,
      pprev,
      ptid,
      clast,
      clen,
      clist,
      beam,
      ldbeam,
      ldseq_len,
      blid,
      bs,
      score,
      block_sort_smem_size);

  return 0;
}

int CTC_prob_merge_V2(
    LogProb* log_prob_struct,
    int step,
    float* ptable,
    float* ptablen,
    int* ptid,
    int* clast,
    int* clist,
    int* clen,
    int lc,
    int ldc,
    int beam,
    int ldbeam,
    int ldseq_len,
    int bs,
    cudaStream_t stream,
    int blid) {
  dim3 grid, block;
  int smem;
  block.x = ldbeam, block.y = 1, block.z = 1;
  grid.x = beam, grid.y = bs, grid.z = 1;

  smem = 0;
  matrix_merge_kernel_v2<<<grid, block, smem, stream>>>(
      (*log_prob_struct),
      step,
      ptable,
      ptablen,
      ptid,
      clast,
      clist,
      clen,
      lc,
      ldc,
      beam,
      ldbeam,
      ldseq_len,
      bs,
      blid);
  return 0;
}

int CTC_prob_topK_V2(
    LogProb* log_prob_struct,
    int step,
    float2* pprev,
    float* ptable,
    float* ptablen,
    int* ptid,
    int* clast,
    int* clen,
    int* clen2,
    int* clist,
    int* clist2,
    int lc,
    int ldc,
    int beam,
    int ldbeam,
    int ldseq_len,
    int blid,
    int bs,
    float* score,
    float* topk_key_buff,
    int* topk_value_buff,
    cudaStream_t stream,
    bool is_last_step) {
  CHECK(beam <= 128, "ERROR: only support beam size <=128 ");

  int all_items_per_batch = ldc * beam;
  constexpr int items_per_thread0 = 4;
  // #define USE_BLOCKS_PER_BATCH 4

#ifdef USE_BLOCKS_PER_BATCH
  constexpr int threads_per_block0 = 256;
  constexpr int items_per_block_per_iter0 =
      threads_per_block0 * items_per_thread0;
  int bxs =
      min(USE_BLOCKS_PER_BATCH,
          (all_items_per_batch + items_per_block_per_iter0 - 1) /
              items_per_block_per_iter0);
  CHECK(
      bxs * bs <= MAX_BLOCKS,
      " ERROR: (batch_size * USE_BLOCKS_PER BATCH) should <=MAX_BLOCKS");
#else
  int max_bxs_per_batch = std::max(1, MAX_BLOCKS / bs);
  constexpr int MAX_BLOCKS_PER_BATCH = 16;
  max_bxs_per_batch = std::min(MAX_BLOCKS_PER_BATCH, max_bxs_per_batch);
  constexpr int threads_per_block0 = 128;
  constexpr int items_per_block_per_iter0 =
      threads_per_block0 * items_per_thread0;
  int bxs =
      min(max_bxs_per_batch,
          (all_items_per_batch + items_per_block_per_iter0 - 1) /
              items_per_block_per_iter0);
#endif
  dim3 grid(bxs, bs);
  dim3 block(threads_per_block0);
  FastDivmod ldc_fast_div{ldc};

  constexpr int Capacity = 32; // 8,16,32,64,128

  using FunType = decltype(bitonic_topk_multi_block_per_batch_kernel<
                           threads_per_block0,
                           Capacity>);
  static FunType* BitonicTopkFuns[5]{
      bitonic_topk_multi_block_per_batch_kernel<threads_per_block0, 8>,
      bitonic_topk_multi_block_per_batch_kernel<threads_per_block0, 16>,
      bitonic_topk_multi_block_per_batch_kernel<threads_per_block0, 32>,
      bitonic_topk_multi_block_per_batch_kernel<threads_per_block0, 64>,
      bitonic_topk_multi_block_per_batch_kernel<threads_per_block0, 128>};
  int need_capacity = topk::calc_capacity(beam);
  int fun_idx = 0;
  fun_idx = std::max(0, 31 - clz(need_capacity) - 3);
  int actual_capacity = (1 << (fun_idx + 3));
  int num_of_subwarp = threads_per_block0 / std::min<int>(32, actual_capacity);
  int smem_size = cu_ctc::topk::calc_smem_size_for_block_wide<float, int>(
      num_of_subwarp, beam);
  auto kernel = BitonicTopkFuns[fun_idx];
  kernel<<<grid, block, smem_size, stream>>>(
      (*log_prob_struct),
      step,
      ptable,
      ptablen,
      lc,
      ldc,
      beam,
      bs,
      topk_key_buff,
      topk_value_buff,
      ldc_fast_div);

  constexpr int threads_per_block1 = 128;
  constexpr int items_per_thread1 = 2;
  const int items_per_batch = bxs * beam;

  topk_reduce_and_copy_list_per_batch_kernel<
      threads_per_block1,
      items_per_thread1><<<bs, threads_per_block1, 0, stream>>>(
      (*log_prob_struct),
      step,
      beam,
      items_per_batch,
      bs,
      topk_key_buff,
      topk_value_buff,
      ldc,
      ldbeam,
      ldseq_len,
      pprev,
      ptable,
      ptablen,
      clast,
      clen,
      clen2,
      clist,
      clist2,
      blid,
      score);
  return 0;
};

template <int BLOCK_SIZE, int ITEMS_PT>
__global__ void init_log_prob_select_kernel(
    LogProb log_prob_struct,
    int blid,
    float threshold) {
  // select seqs that log_prob[blid]< threshold
  int batch_id = blockIdx.x;

  using BlockScanT = cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  int selected[ITEMS_PT];
  int selected_scan[ITEMS_PT];
#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PT; ITEM++) {
    selected[ITEM] = 0;
  }
  const int tx = threadIdx.x;
  int this_batch_seq_len = log_prob_struct.origin_seq_lens[batch_id];
  int block_agg = 0;
  for (int seq_id_offset = 0; seq_id_offset < this_batch_seq_len;
       seq_id_offset += (BLOCK_SIZE * ITEMS_PT)) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PT; ITEM++) {
      int seq_id = seq_id_offset + ITEMS_PT * tx + ITEM;
      if (seq_id < this_batch_seq_len) {
        selected[ITEM] =
            (log_prob_struct.at(batch_id, seq_id, blid) < threshold) ? 1 : 0;
      } else {
        selected[ITEM] = 0;
      }
    }
    __syncthreads();
    int block_agg_this_iter = 0;
    BlockScanT{temp_storage}.ExclusiveSum(
        selected, selected_scan, block_agg_this_iter);
    __syncthreads();

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PT; ITEM++) {
      int seq_id = seq_id_offset + ITEMS_PT * tx + ITEM;
      if (selected[ITEM]) {
        log_prob_struct.select_seqs
            [batch_id * log_prob_struct.seq_len + selected_scan[ITEM] +
             block_agg] = seq_id;
      }
    }

    block_agg += block_agg_this_iter;
  }
  if (tx == 0) {
    log_prob_struct.select_seq_lens[batch_id] = block_agg;
  }
}

int init_log_prob_and_cal_max_select_seq_len(
    LogProb* log_prob_struct,
    int blid,
    float threshold,
    cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int ITEMS_PT = 4;
  int bxs = log_prob_struct->batch;
  init_log_prob_select_kernel<BLOCK_SIZE, ITEMS_PT>
      <<<bxs, BLOCK_SIZE, 0, stream>>>((*log_prob_struct), blid, threshold);

  // for simplicity ,  find max_select_seq_len on cpu
  std::vector<int> select_seq_lens(bxs);
  CUDA_CHECK(cudaMemcpyAsync(
      select_seq_lens.data(),
      log_prob_struct->select_seq_lens,
      sizeof(int) * bxs,
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int ret_max_select_seq_len =
      *std::max_element(select_seq_lens.begin(), select_seq_lens.end());

  return ret_max_select_seq_len;
}

// if the parity of select_seq_len is different from the max_select_seq_len,
// their clist and clen need to be copy to another clist and clen
template <int SUB_WARP_SIZE, int BLOCK_SIZE>
__global__ void copy_list_len_for_diff_parity_kernel(
    LogProb log_prob_struct,
    int step,
    int max_select_seq_len,
    int* clen,
    int* clen2,
    int* clist,
    int* clist2,
    int bs,
    int beam,
    int ldbeam,
    int ldseq_len) {
  const int batch_id = blockIdx.y;
  if (batch_id >= bs)
    return;
  int select_seq_len = log_prob_struct.select_seq_lens[batch_id];
  if ((select_seq_len & 1) == (max_select_seq_len & 1))
    return;
  const int bx = blockIdx.x;

  constexpr int beams_per_block = BLOCK_SIZE / SUB_WARP_SIZE;
  const int tx = threadIdx.x;
  const int sub_warp_id = tx / SUB_WARP_SIZE;
  const int tid_in_sub_warp = tx % SUB_WARP_SIZE;
  const int beamid = bx * beams_per_block + sub_warp_id;
  if (beamid >= beam)
    return;

  int new_len = clen[batch_id * ldbeam + beamid];
  if (tid_in_sub_warp == 0) {
    clen2[batch_id * ldbeam + beamid] = new_len;
  }
  for (int id = tid_in_sub_warp; id < new_len; id += SUB_WARP_SIZE) {
    clist2[batch_id * beam * ldseq_len + beamid * ldseq_len + id] =
        clist[batch_id * beam * ldseq_len + beamid * ldseq_len + id];
  }
}

__global__ void copy_list_len_for_diff_parity_simple_kernel(
    LogProb log_prob_struct,
    int step,
    int max_select_seq_len,
    int* clen,
    int* clen2,
    int* clist,
    int* clist2,
    int bs,
    int beam,
    int ldbeam,
    int ldseq_len) {
  const int batch_id = blockIdx.x;
  if (batch_id >= bs)
    return;
  int select_seq_len = log_prob_struct.select_seq_lens[batch_id];
  if ((select_seq_len & 1) == (max_select_seq_len & 1))
    return;
  const int tx = threadIdx.x;
  for (int beamid = tx; beamid < beam; beamid += blockDim.x) {
    int new_len = clen[batch_id * ldbeam + beamid];
    clen2[batch_id * ldbeam + beamid] = new_len;
    for (int i = 0; i < new_len; i++) {
      clist2[batch_id * beam * ldseq_len + beamid * ldseq_len + i] =
          clist[batch_id * beam * ldseq_len + beamid * ldseq_len + i];
    }
  }
}
int CTC_copy_list_len_for_differnet_parity(
    LogProb* log_prob_struct,
    int step,
    int max_select_seq_len,
    int* clen,
    int* clen2,
    int* clist,
    int* clist2,
    int bs,
    int beam,
    int ldbeam,
    int ldseq_len,
    cudaStream_t stream) {
  constexpr int SUB_WARP_SIZE = 8;
  constexpr int BLOCK_SIZE = 256;
  const int beams_per_block = BLOCK_SIZE / SUB_WARP_SIZE;
  const int bxs = (beam + beams_per_block - 1) / beams_per_block;
  dim3 blocks_this_grid;
  blocks_this_grid.x = bxs;
  blocks_this_grid.y = bs;
  blocks_this_grid.z = 1;
  copy_list_len_for_diff_parity_kernel<SUB_WARP_SIZE, BLOCK_SIZE>
      <<<blocks_this_grid, BLOCK_SIZE, 0, stream>>>(
          (*log_prob_struct),
          step,
          max_select_seq_len,
          clen,
          clen2,
          clist,
          clist2,
          bs,
          beam,
          ldbeam,
          ldseq_len);

  return 0;
}
} // namespace cu_ctc
