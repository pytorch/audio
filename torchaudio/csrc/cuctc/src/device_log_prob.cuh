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
#pragma once
namespace cu_ctc {
struct LogProb {
  float* data_ptr;
  int batch;
  int seq_len;
  int vocab_size;
  int batch_stride;
  int seq_len_stride;
  int vocab_stride;
  int* origin_seq_lens; // batchs
  int* select_seqs; // batchs *seq_len;
  int* select_seq_lens; // batchs

  __device__ __forceinline__ float at(int batch_id, int seq_id, int char_id) {
    return data_ptr
        [batch_id * batch_stride + seq_id * seq_len_stride +
         char_id * vocab_stride];
  }
  __device__ __forceinline__ int ith_selected_seq_in_this_batch(
      int batch_id,
      int i) {
    return select_seqs[batch_id * seq_len + i];
  }
  __device__ __forceinline__ bool need_process_on_ith_step(
      int batch_id,
      int istep) {
    return istep < select_seq_lens[batch_id];
  }

  /**
   * @brief if the prob of blank in  next original timestep > threshold , we
   * will not process the next original timestep, but will process the
   * subsequent blank on the currently processed timestep.
   *
   * @param batch_id
   * @param istep
   * @return __device__
   */
  __device__ __forceinline__ bool need_add_blank(int batch_id, int istep) {
    if ((istep < 0) || (istep + 1) >= select_seq_lens[batch_id]) {
      return false;
    }
    if ((ith_selected_seq_in_this_batch(batch_id, istep + 1) -
         ith_selected_seq_in_this_batch(batch_id, istep)) > 1) {
      return true;
    }
    return false;
  }
};

} // namespace cu_ctc
