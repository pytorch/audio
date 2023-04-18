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
#ifndef __ctc_prefix_decoder_h_
#define __ctc_prefix_decoder_h_

#include <cuda_runtime.h>

#include <tuple>
#include <vector>
namespace cu_ctc {

struct InternalData;
std::uintptr_t prefixCTC_alloc(std::uintptr_t stream_ptr);
void prefixCTC_free(std::uintptr_t inter_data_ptr);

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
    float threshold);
int ctc_beam_search_decoder_batch_gpu(
    InternalData* inter_data,
    float* pp,
    int blid,
    int spid,
    int* clist,
    int* clen,
    float* score);

} // namespace cu_ctc

#endif
