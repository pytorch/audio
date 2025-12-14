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

#include <libtorchaudio/utils.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <tuple>
#include <utility>
#include <vector>
#include "include/ctc_prefix_decoder.h"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

std::tuple<size_t, Tensor, Tensor, Tensor> ctc_prefix_decoder_batch_wrapper(
    std::uintptr_t n_inter_data,
    Tensor buff,
    Tensor log_prob,
    Tensor encoder_out_lens,
    int beam,
    int blid,
    int spid,
    float thresold) {
  STD_TORCH_CHECK(
      encoder_out_lens.scalar_type() == ScalarType::Int,
      "encoder_out_lens must be torch.int32");
  STD_TORCH_CHECK(
      log_prob.scalar_type() == ScalarType::Float,
      "log_prob must be torch.float32");
  STD_TORCH_CHECK(log_prob.is_cuda(), "log_prob must be cuda tensor");
  STD_TORCH_CHECK(
      encoder_out_lens.is_cuda(), "encoder_out_lens must be cuda tensor");
  STD_TORCH_CHECK(
      log_prob.get_device_index() == encoder_out_lens.get_device_index(),
      "log_prob and encoder_out_lens must be on the same device");
  STD_TORCH_CHECK(log_prob.is_contiguous(), "log_prob must be contiguous");
  STD_TORCH_CHECK(
      encoder_out_lens.is_contiguous(), "encoder_out_lens must be contiguous");
  if (buff.numel() > 0) {
    STD_TORCH_CHECK(buff.is_cuda(), "buff must be cuda tensor");
    STD_TORCH_CHECK(
        log_prob.get_device_index() == buff.get_device_index(),
        "log_prob and buff must be on the same device");
  }
  auto pp_sizes_ = log_prob.sizes();
  auto pp_strides_ = log_prob.strides();
  std::vector<int> pp_sizes(std::begin(pp_sizes_), std::end(pp_sizes_));
  std::vector<int> pp_strides(std::begin(pp_strides_), std::end(pp_strides_));
  cu_ctc::InternalData* inter_data = (cu_ctc::InternalData*)(n_inter_data);
  auto [require_size, max_select_seq_len] =
      cu_ctc::calculate_require_buff_and_init_internal_data(
          inter_data,
          pp_sizes[0],
          pp_sizes[1],
          pp_sizes[2],
          beam,
          reinterpret_cast<std::uintptr_t>(buff.data_ptr()),
          buff.size(0),
          log_prob.mutable_data_ptr<float>(),
          encoder_out_lens.mutable_data_ptr<int>(),
          pp_sizes,
          pp_strides,
          blid,
          thresold);
  if (require_size > 0) {
    return std::make_tuple(require_size, Tensor{}, Tensor{}, Tensor{});
  }
  int batch_size = pp_sizes[0];
  Tensor list_data = torch::stable::empty(
      {batch_size, beam, max_select_seq_len}, ScalarType::Int);
  Tensor len_data = torch::stable::empty({batch_size, beam}, ScalarType::Int);
  Tensor score = torch::stable::empty({batch_size, beam}, ScalarType::Float);
  cu_ctc::ctc_beam_search_decoder_batch_gpu(
      inter_data,
      blid,
      spid,
      list_data.mutable_data_ptr<int>(),
      len_data.mutable_data_ptr<int>(),
      score.mutable_data_ptr<float>());
  return std::make_tuple(
      require_size,
      std::move(list_data),
      std::move(len_data),
      std::move(score));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio_prefixctc, CompositeExplicitAutograd, m) {
  m.impl("prefixCTC_alloc", TORCH_BOX(&cu_ctc::prefixCTC_alloc));
  m.impl("prefixCTC_free", TORCH_BOX(&cu_ctc::prefixCTC_free));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio_prefixctc, CUDA, m) {
  m.impl(
      "ctc_beam_search_decoder_batch_gpu_v2",
      TORCH_BOX(&ctc_prefix_decoder_batch_wrapper));
}

STABLE_TORCH_LIBRARY(torchaudio_prefixctc, m) {
  m.def(
      "ctc_beam_search_decoder_batch_gpu_v2(int interal_data_ptr, Tensor memory, Tensor log_prob, Tensor encoder_out_lens, int beam, int blid, int spid, float thresold) -> (int, Tensor, Tensor, Tensor)");
  m.def("prefixCTC_alloc(int stream) -> int");
  m.def("prefixCTC_free(int interal_data_ptr) -> ()");
}

// Defines PyInit_torchaudio_prefixctc when building under Windows:
TORCHAUDIO_EXT_MODULE(torchaudio_prefixctc)
