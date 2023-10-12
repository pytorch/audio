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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <utility>
#include <vector>
#include "include/ctc_prefix_decoder.h"
namespace py = pybind11;

std::tuple<size_t, std::vector<std::vector<std::pair<float, std::vector<int>>>>>
ctc_prefix_decoder_batch_wrapper(
    std::uintptr_t n_inter_data,
    std::uintptr_t buff_ptr,
    size_t buff_size,
    std::uintptr_t pp,
    std::uintptr_t seq_len_ptr,
    const std::vector<int>& pp_sizes,
    const std::vector<int>& pp_strides,
    int beam,
    int blid,
    int spid,
    float thresold) {
  using SCORE_TYPE =
      std::vector<std::vector<std::pair<float, std::vector<int>>>>;
  cu_ctc::InternalData* inter_data = (cu_ctc::InternalData*)(n_inter_data);
  auto [require_size, max_select_seq_len] =
      cu_ctc::calculate_require_buff_and_init_internal_data(
          inter_data,
          pp_sizes[0],
          pp_sizes[1],
          pp_sizes[2],
          beam,
          buff_ptr,
          buff_size,
          (float*)pp,
          (int*)seq_len_ptr,
          pp_sizes,
          pp_strides,
          blid,
          thresold);
  if (require_size > 0) {
    return std::make_tuple(require_size, SCORE_TYPE{});
  }
  int batch_size = pp_sizes[0];
  std::vector<int> list_data(batch_size * beam * max_select_seq_len);
  std::vector<int> len_data(batch_size * beam);
  std::vector<float> score(batch_size * beam);
  cu_ctc::ctc_beam_search_decoder_batch_gpu(
      inter_data, blid, spid, list_data.data(), len_data.data(), score.data());
  SCORE_TYPE score_hyps{};
  score_hyps.reserve(batch_size);
  for (int b = 0; b < batch_size; b++) {
    score_hyps.push_back(std::vector<std::pair<float, std::vector<int>>>{});
    score_hyps.back().reserve(beam);
    for (int beam_id = 0; beam_id < beam; beam_id++) {
      int len = len_data[b * beam + beam_id];
      int offset = b * beam * max_select_seq_len + beam_id * max_select_seq_len;
      std::vector<int> clist(
          list_data.data() + offset, list_data.data() + offset + len);
      score_hyps.back().push_back(
          std::pair{score[b * beam + beam_id], std::move(clist)});
    }
  }
  return std::make_tuple(require_size, std::move(score_hyps));
}

PYBIND11_MODULE(pybind11_prefixctc, m) {
  m.doc() = "none";
  m.def(
      "ctc_beam_search_decoder_batch_gpu_v2",
      &ctc_prefix_decoder_batch_wrapper,
      "ctc prefix decoder  v2 computing on GPU");
  m.def("prefixCTC_alloc", &cu_ctc::prefixCTC_alloc, "allocate internal data");
  m.def("prefixCTC_free", &cu_ctc::prefixCTC_free, "free internal data");
}
