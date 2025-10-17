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
#include <cuda_runtime.h>

#include "../include/ctc_prefix_decoder.h"
#include "../include/ctc_prefix_decoder_host.h"

#include "device_data_wrap.h"
#include "device_log_prob.cuh"

namespace cu_ctc {
struct InternalData {
  cudaStream_t stream;

  int lc;
  int ldc;
  int bs;
  int beam;
  int ldbeam;
  int time;
  int ldseq_len;
  DeviceDataWrap<float2> pprev;
  DeviceDataWrap<float> ptable;
  DeviceDataWrap<float> ptablen;
  DeviceDataWrap<int> clast;
  DeviceDataWrap<int> clen[2];
  DeviceDataWrap<int> clist[2];
  DeviceDataWrap<int> ptid;
  DeviceDataWrap<float> score;
  DeviceDataWrap<float> topk_key_buffer;
  DeviceDataWrap<int> topk_value_buffer;
  DeviceDataWrap<int> select_seqs;
  DeviceDataWrap<int> select_seq_lens;
  LogProb log_prob;
  int max_select_seq_len;
};

std::tuple<size_t, int> calculate_require_buff_and_init_internal_data(
    InternalData* inter_data,
    int batch_size,
    int seq_len,
    int vocab_size,
    int beam,
    std::uintptr_t buff_ptr,
    size_t buff_size,
    float* log_prob_data_ptr,
    int* original_lens,
    const std::vector<int>& prob_sizes,
    const std::vector<int>& prob_strides,
    int blid,
    float threshold) {
  if ((batch_size * beam * seq_len * vocab_size) <= 0) {
    return {0, 0};
  }

  CHECK(prob_sizes.size() == 3, "only support 3D log_prob.");
  CHECK(prob_strides.size() == 3, "only support 3D log_prob. ");
  CHECK(
      prob_sizes[0] == batch_size && prob_sizes[1] == seq_len &&
          prob_sizes[2] == vocab_size,
      "batch_size ,seq_len ,vocab_size must match with porb_size");

  auto align_size = [](size_t size) -> size_t {
    return (size + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
  };
  int lc = vocab_size;
  int ldc = lc;
  int ldbeam = ((beam - 1) / 16 + 1) * 16;
  int ldseq_len = (seq_len + 16 - 1) / 16 * 16;
  int bs = batch_size;
  int time = seq_len;
  size_t require_size = 0;

  size_t pprev_size = sizeof(float2) * bs * ldbeam;
  size_t pprev_align_size = align_size(pprev_size);
  require_size += pprev_align_size;
  size_t ptable_size = sizeof(float) * (bs * beam * ldc);
  size_t ptablen_size = sizeof(float) * bs * beam * ldc;
  size_t ptable_align_size = align_size(ptable_size);
  size_t ptablen_align_size = align_size(ptablen_size);
  require_size += ptable_align_size;
  require_size += ptablen_align_size;
  size_t clast_align_size = align_size(sizeof(int) * ldbeam * bs);
  require_size += clast_align_size;
  size_t clen_align_size = align_size(sizeof(int) * ldbeam * bs);
  size_t clist_align_size = align_size(sizeof(int) * ldseq_len * beam * bs);
  require_size += 2 * clen_align_size;
  require_size += 2 * clist_align_size;
  size_t ptid_align_size = align_size(sizeof(int) * bs * ldbeam);
  require_size += ptid_align_size;
  size_t score_align_size = align_size(sizeof(float) * bs * ldbeam);
  require_size += score_align_size;
  size_t key_buff_align_size = align_size(sizeof(float) * beam * MAX_BLOCKS);
  size_t value_buff_align_size = align_size(sizeof(int) * beam * MAX_BLOCKS);
  require_size += (key_buff_align_size + value_buff_align_size);

  size_t select_seqs_align_size =
      align_size(sizeof(int) * batch_size * seq_len);
  require_size += select_seqs_align_size;
  size_t select_seq_lens_align_size = align_size(sizeof(int) * batch_size);
  require_size += select_seq_lens_align_size;

  require_size += ALIGN_BYTES;
  if (require_size > buff_size) {
    return {require_size, 0};
  }

  char* buff_align_ptr = reinterpret_cast<char*>(align_size(buff_ptr));

  inter_data->beam = beam;
  inter_data->ldbeam = ldbeam;
  inter_data->bs = bs;
  inter_data->lc = lc;
  inter_data->ldc = ldc;
  inter_data->time = time;
  inter_data->ldseq_len = ldseq_len;

#define SET_DATA(NAME, TYPE, SIZE)                                         \
  inter_data->NAME =                                                       \
      DeviceDataWrap<TYPE>(reinterpret_cast<TYPE*>(buff_align_ptr), SIZE); \
  buff_align_ptr += SIZE;

  SET_DATA(pprev, float2, pprev_align_size);
  SET_DATA(ptable, float, ptable_align_size);
  SET_DATA(ptablen, float, ptable_align_size);
  SET_DATA(clast, int, clast_align_size);
  SET_DATA(clen[0], int, clen_align_size);
  SET_DATA(clen[1], int, clen_align_size);
  SET_DATA(clist[0], int, clist_align_size);
  SET_DATA(clist[1], int, clist_align_size);
  SET_DATA(ptid, int, ptid_align_size);
  SET_DATA(score, float, score_align_size);
  SET_DATA(topk_key_buffer, float, key_buff_align_size);
  SET_DATA(topk_value_buffer, int, value_buff_align_size);
  SET_DATA(select_seqs, int, select_seqs_align_size);
  SET_DATA(select_seq_lens, int, select_seq_lens_align_size);
#undef SET_DATA

  // init log_prob
  inter_data->log_prob.data_ptr = log_prob_data_ptr;
  inter_data->log_prob.origin_seq_lens = original_lens;
  inter_data->log_prob.select_seqs = inter_data->select_seqs.data_ptr();
  inter_data->log_prob.select_seq_lens = inter_data->select_seq_lens.data_ptr();
  inter_data->log_prob.batch = batch_size;
  inter_data->log_prob.vocab_size = vocab_size;
  inter_data->log_prob.seq_len = seq_len;
  inter_data->log_prob.batch_stride = prob_strides[0];
  inter_data->log_prob.seq_len_stride = prob_strides[1];
  inter_data->log_prob.vocab_stride = prob_strides[2];

  inter_data->max_select_seq_len = init_log_prob_and_cal_max_select_seq_len(
      &(inter_data->log_prob), blid, threshold, inter_data->stream);
  return {0, inter_data->max_select_seq_len};
}

int prefixCTC_V2(
    InternalData* inter_data,
    int blid,
    int spid,
    int step,
    bool is_last_step,
    int max_select_seq_len) {
  LogProb* log_prob_struct = &(inter_data->log_prob);
  if (step == 0) {
    CTC_prob_first_step_V2(
        log_prob_struct,
        step,
        inter_data->pprev,
        inter_data->ptid,
        inter_data->clast,
        inter_data->clen[step % 2],
        inter_data->clist[step % 2],
        inter_data->beam,
        inter_data->ldbeam,
        inter_data->ldseq_len,
        inter_data->bs,
        inter_data->score,
        inter_data->stream,
        blid);
  } else {
    CTC_prob_matrix_V2(
        log_prob_struct,
        step,
        inter_data->pprev,
        inter_data->ptable,
        inter_data->ptablen,
        inter_data->clast,
        inter_data->lc,
        inter_data->ldc,
        inter_data->beam,
        inter_data->ldbeam,
        inter_data->bs,
        blid,
        spid,
        inter_data->stream);
    CTC_prob_merge_V2(
        log_prob_struct,
        step,
        inter_data->ptable,
        inter_data->ptablen,
        inter_data->ptid,
        inter_data->clast,
        inter_data->clist[(step % 2) ^ 1],
        inter_data->clen[(step % 2) ^ 1],
        inter_data->lc,
        inter_data->ldc,
        inter_data->beam,
        inter_data->ldbeam,
        inter_data->ldseq_len,
        inter_data->bs,
        inter_data->stream,
        blid);
    CTC_prob_topK_V2(
        log_prob_struct,
        step,
        inter_data->pprev,
        inter_data->ptable,
        inter_data->ptablen,
        inter_data->ptid,
        inter_data->clast,
        inter_data->clen[(step % 2) ^ 1],
        inter_data->clen[(step % 2)],
        inter_data->clist[(step % 2) ^ 1],
        inter_data->clist[(step % 2)],
        inter_data->lc,
        inter_data->ldc,
        inter_data->beam,
        inter_data->ldbeam,
        inter_data->ldseq_len,
        blid,
        inter_data->bs,
        inter_data->score,
        inter_data->topk_key_buffer,
        inter_data->topk_value_buffer,
        inter_data->stream,
        is_last_step);
    if (is_last_step) {
      // if the parity of select_seq_len is different from the
      // max_select_seq_len, their clist and clen need to be copy to another
      // clist and clen
      CTC_copy_list_len_for_differnet_parity(
          log_prob_struct,
          step,
          max_select_seq_len,
          inter_data->clen[(step % 2) ^ 1],
          inter_data->clen[(step % 2)],
          inter_data->clist[(step % 2) ^ 1],
          inter_data->clist[(step % 2)],
          inter_data->bs,
          inter_data->beam,
          inter_data->ldbeam,
          inter_data->ldseq_len,
          inter_data->stream);
    }
  }
  return 0;
}

std::uintptr_t prefixCTC_alloc(std::uintptr_t stream_ptr) {
  InternalData* Inter_data = new InternalData;
  Inter_data->stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  return reinterpret_cast<std::uintptr_t>(Inter_data);
}

void prefixCTC_free(std::uintptr_t inter_data_ptr) {
  InternalData* inter_data = reinterpret_cast<InternalData*>(inter_data_ptr);
  delete inter_data;
}

int ctc_beam_search_decoder_batch_gpu(
    InternalData* inter_data,
    int blid,
    int spid,
    int* clist,
    int* clen,
    float* score) {
  // batch_pprev: time x batch x lc
  // internal_data *data = (internal_data *)data_int;

  CUDA_CHECK(cudaMemsetAsync(
      (inter_data->clast.data_ptr()),
      0,
      inter_data->clast.size_in_byte(),
      inter_data->stream));
  CUDA_CHECK(cudaMemsetAsync(
      (inter_data->clen[0].data_ptr()),
      0,
      inter_data->clen[0].size_in_byte(),
      inter_data->stream));
  CUDA_CHECK(cudaMemsetAsync(
      (inter_data->clen[1].data_ptr()),
      0,
      inter_data->clen[0].size_in_byte(),
      inter_data->stream));
  CUDA_CHECK(cudaMemsetAsync(
      (inter_data->clist[0].data_ptr()),
      -1,
      inter_data->clen[0].size_in_byte(),
      inter_data->stream));
  CUDA_CHECK(cudaMemsetAsync(
      (inter_data->clist[1].data_ptr()),
      -1,
      inter_data->clen[0].size_in_byte(),
      inter_data->stream));

  // ptable  the table of prob for end_in_bank (bs*beam*vocab_size)
  // ptablen the table of prob for no_end_in_bank(ba*beam*vocab_size)
  int step = 0;
  while (step < inter_data->max_select_seq_len) {
    bool is_last_step = (step == (inter_data->max_select_seq_len - 1));

    prefixCTC_V2(
        inter_data,
        blid,
        spid,
        step,
        is_last_step,
        inter_data->max_select_seq_len);
    step++;
  }

  CUDA_CHECK(cudaMemcpy2DAsync(
      clen,
      sizeof(int) * inter_data->beam,
      inter_data->clen[(step % 2) ^ 1].data_ptr(),
      sizeof(int) * inter_data->ldbeam,
      sizeof(int) * inter_data->beam,
      inter_data->bs,
      cudaMemcpyDeviceToHost,
      inter_data->stream));
  CUDA_CHECK(cudaMemcpy2DAsync(
      clist,
      sizeof(int) * inter_data->max_select_seq_len,
      inter_data->clist[(step % 2) ^ 1].data_ptr(),
      sizeof(int) * inter_data->ldseq_len,
      sizeof(int) * inter_data->max_select_seq_len,
      inter_data->beam * inter_data->bs,
      cudaMemcpyDeviceToHost,
      inter_data->stream));

  CUDA_CHECK(cudaMemcpy2DAsync(
      score,
      sizeof(float) * inter_data->beam,
      inter_data->score.data_ptr(),
      sizeof(float) * inter_data->ldbeam,
      sizeof(float) * inter_data->beam,
      inter_data->bs,
      cudaMemcpyDeviceToHost,
      inter_data->stream));

  CUDA_CHECK(cudaStreamSynchronize(inter_data->stream));

  return 0;
}

} // namespace cu_ctc
