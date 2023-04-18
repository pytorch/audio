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
#ifndef __ctc_prefix_decoder_host_h_
#define __ctc_prefix_decoder_host_h_

#include <cuda_runtime.h>

#define CUDA_CHECK(X)                                   \
  do {                                                  \
    auto result = X;                                    \
    if (result != cudaSuccess) {                        \
      const char* p_err_str = cudaGetErrorName(result); \
      fprintf(                                          \
          stderr,                                       \
          "File %s Line %d %s returned %s.\n",          \
          __FILE__,                                     \
          __LINE__,                                     \
          #X,                                           \
          p_err_str);                                   \
      abort();                                          \
    }                                                   \
  } while (0)

#define CHECK(X, ERROR_INFO)                        \
  do {                                              \
    auto result = (X);                              \
    if (!result) {                                  \
      fprintf(                                      \
          stderr,                                   \
          " File %s Line %d %s ERROR_INFO: %s .\n", \
          __FILE__,                                 \
          __LINE__,                                 \
          #X,                                       \
          ERROR_INFO);                              \
      abort();                                      \
    }                                               \
  } while (0)

namespace cu_ctc {

struct LogProb;
int init_log_prob_and_cal_max_select_seq_len(
    LogProb* log_prob_struct,
    int blid,
    float threshold,
    cudaStream_t stream);

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
    cudaStream_t stream);
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
    int blid);

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
    int blid);
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
    bool is_last_step);
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
    cudaStream_t stream);

} // namespace cu_ctc

#endif
