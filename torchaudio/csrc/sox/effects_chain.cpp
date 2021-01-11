#include <torchaudio/csrc/sox/effects_chain.h>
#include <torchaudio/csrc/sox/utils.h>

using namespace torch::indexing;
using namespace torchaudio::sox_utils;

namespace torchaudio {
namespace sox_effects_chain {

namespace {

// Helper struct to safely close sox_effect_t* pointer returned by
// sox_create_effect
struct SoxEffect {
  explicit SoxEffect(sox_effect_t* se) noexcept : se_(se){};
  SoxEffect(const SoxEffect& other) = delete;
  SoxEffect(const SoxEffect&& other) = delete;
  SoxEffect& operator=(const SoxEffect& other) = delete;
  SoxEffect& operator=(SoxEffect&& other) = delete;
  ~SoxEffect() {
    if (se_ != nullptr) {
      free(se_);
    }
  }
  operator sox_effect_t*() const {
    return se_;
  };
  sox_effect_t* operator->() noexcept {
    return se_;
  }

 private:
  sox_effect_t* se_;
};

/// helper classes for passing the location of input tensor and output buffer
///
/// drain/flow callback functions require plaing C style function signature and
/// the way to pass extra data is to attach data to sox_fffect_t::priv pointer.
/// The following structs will be assigned to sox_fffect_t::priv pointer which
/// gives sox_effect_t an access to input Tensor and output buffer object.
struct TensorInputPriv {
  size_t index;
  TensorSignal* signal;
};
struct TensorOutputPriv {
  std::vector<sox_sample_t>* buffer;
};
struct FileOutputPriv {
  sox_format_t* sf;
};

/// Callback function to feed Tensor data to SoxEffectChain.
int tensor_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  // Retrieve the input Tensor and current index
  auto priv = static_cast<TensorInputPriv*>(effp->priv);
  auto index = priv->index;
  auto signal = priv->signal;
  auto tensor = signal->getTensor();
  auto num_channels = effp->out_signal.channels;

  // Adjust the number of samples to read
  const size_t num_samples = tensor.numel();
  if (index + *osamp > num_samples) {
    *osamp = num_samples - index;
  }
  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % num_channels;

  // Slice the input Tensor and unnormalize the values
  const auto tensor_ = [&]() {
    auto i_frame = index / num_channels;
    auto num_frames = *osamp / num_channels;
    auto t = (signal->getChannelsFirst())
        ? tensor.index({Slice(), Slice(i_frame, i_frame + num_frames)}).t()
        : tensor.index({Slice(i_frame, i_frame + num_frames), Slice()});
    return unnormalize_wav(t.reshape({-1})).contiguous();
  }();
  priv->index += *osamp;

  // Write data to SoxEffectsChain buffer.
  auto ptr = tensor_.data_ptr<int32_t>();
  std::copy(ptr, ptr + *osamp, obuf);

  return (priv->index == num_samples) ? SOX_EOF : SOX_SUCCESS;
}

/// Callback function to fetch data from SoxEffectChain.
int tensor_output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  *osamp = 0;
  // Get output buffer
  auto out_buffer = static_cast<TensorOutputPriv*>(effp->priv)->buffer;
  // Append at the end
  out_buffer->insert(out_buffer->end(), ibuf, ibuf + *isamp);
  return SOX_SUCCESS;
}

int file_output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  *osamp = 0;
  if (*isamp) {
    auto sf = static_cast<FileOutputPriv*>(effp->priv)->sf;
    if (sox_write(sf, ibuf, *isamp) != *isamp) {
      if (sf->sox_errno) {
        std::ostringstream stream;
        stream << sf->sox_errstr << " " << sox_strerror(sf->sox_errno) << " "
               << sf->filename;
        throw std::runtime_error(stream.str());
      }
      return SOX_EOF;
    }
  }
  return SOX_SUCCESS;
}

sox_effect_handler_t* get_tensor_input_handler() {
  static sox_effect_handler_t handler{/*name=*/"input_tensor",
                                      /*usage=*/NULL,
                                      /*flags=*/SOX_EFF_MCHAN,
                                      /*getopts=*/NULL,
                                      /*start=*/NULL,
                                      /*flow=*/NULL,
                                      /*drain=*/tensor_input_drain,
                                      /*stop=*/NULL,
                                      /*kill=*/NULL,
                                      /*priv_size=*/sizeof(TensorInputPriv)};
  return &handler;
}

sox_effect_handler_t* get_tensor_output_handler() {
  static sox_effect_handler_t handler{/*name=*/"output_tensor",
                                      /*usage=*/NULL,
                                      /*flags=*/SOX_EFF_MCHAN,
                                      /*getopts=*/NULL,
                                      /*start=*/NULL,
                                      /*flow=*/tensor_output_flow,
                                      /*drain=*/NULL,
                                      /*stop=*/NULL,
                                      /*kill=*/NULL,
                                      /*priv_size=*/sizeof(TensorOutputPriv)};
  return &handler;
}

sox_effect_handler_t* get_file_output_handler() {
  static sox_effect_handler_t handler{/*name=*/"output_file",
                                      /*usage=*/NULL,
                                      /*flags=*/SOX_EFF_MCHAN,
                                      /*getopts=*/NULL,
                                      /*start=*/NULL,
                                      /*flow=*/file_output_flow,
                                      /*drain=*/NULL,
                                      /*stop=*/NULL,
                                      /*kill=*/NULL,
                                      /*priv_size=*/sizeof(FileOutputPriv)};
  return &handler;
}

} // namespace

SoxEffectsChain::SoxEffectsChain(
    sox_encodinginfo_t input_encoding,
    sox_encodinginfo_t output_encoding)
    : in_enc_(input_encoding),
      out_enc_(output_encoding),
      in_sig_(),
      interm_sig_(),
      out_sig_(),
      sec_(sox_create_effects_chain(&in_enc_, &out_enc_)) {
  if (!sec_) {
    throw std::runtime_error("Failed to create effect chain.");
  }
}

SoxEffectsChain::~SoxEffectsChain() {
  if (sec_ != nullptr) {
    sox_delete_effects_chain(sec_);
  }
}

void SoxEffectsChain::run() {
  sox_flow_effects(sec_, NULL, NULL);
}

void SoxEffectsChain::addInputTensor(TensorSignal* signal) {
  in_sig_ = get_signalinfo(signal, "wav");
  interm_sig_ = in_sig_;
  SoxEffect e(sox_create_effect(get_tensor_input_handler()));
  auto priv = static_cast<TensorInputPriv*>(e->priv);
  priv->signal = signal;
  priv->index = 0;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error("Internal Error: Failed to add effect: input_tensor");
  }
}

void SoxEffectsChain::addOutputBuffer(
    std::vector<sox_sample_t>* output_buffer) {
  SoxEffect e(sox_create_effect(get_tensor_output_handler()));
  static_cast<TensorOutputPriv*>(e->priv)->buffer = output_buffer;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error("Internal Error: Failed to add effect: output_tensor");
  }
}

void SoxEffectsChain::addInputFile(sox_format_t* sf) {
  in_sig_ = sf->signal;
  interm_sig_ = in_sig_;
  SoxEffect e(sox_create_effect(sox_find_effect("input")));
  char* opts[] = {(char*)sf};
  sox_effect_options(e, 1, opts);
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Internal Error: Failed to add effect: input " << sf->filename;
    throw std::runtime_error(stream.str());
  }
}

void SoxEffectsChain::addOutputFile(sox_format_t* sf) {
  out_sig_ = sf->signal;
  SoxEffect e(sox_create_effect(get_file_output_handler()));
  static_cast<FileOutputPriv*>(e->priv)->sf = sf;
  if (sox_add_effect(sec_, e, &interm_sig_, &out_sig_) != SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Internal Error: Failed to add effect: output " << sf->filename;
    throw std::runtime_error(stream.str());
  }
}

void SoxEffectsChain::addEffect(const std::vector<std::string> effect) {
  const auto num_args = effect.size();
  if (num_args == 0) {
    throw std::runtime_error("Invalid argument: empty effect.");
  }
  const auto name = effect[0];
  if (UNSUPPORTED_EFFECTS.find(name) != UNSUPPORTED_EFFECTS.end()) {
    std::ostringstream stream;
    stream << "Unsupported effect: " << name;
    throw std::runtime_error(stream.str());
  }

  SoxEffect e(sox_create_effect(sox_find_effect(name.c_str())));
  const auto num_options = num_args - 1;

  std::vector<char*> opts;
  for (size_t i = 1; i < num_args; ++i) {
    opts.push_back((char*)effect[i].c_str());
  }
  if (sox_effect_options(e, num_options, num_options ? opts.data() : nullptr) !=
      SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Invalid effect option:";
    for (const auto& v : effect) {
      stream << " " << v;
    }
    throw std::runtime_error(stream.str());
  }

  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    std::ostringstream stream;
    stream << "Internal Error: Failed to add effect: \"" << name;
    for (size_t i = 1; i < num_args; ++i) {
      stream << " " << effect[i];
    }
    stream << "\"";
    throw std::runtime_error(stream.str());
  }
}

int64_t SoxEffectsChain::getOutputNumChannels() {
  return interm_sig_.channels;
}

int64_t SoxEffectsChain::getOutputSampleRate() {
  return interm_sig_.rate;
}

#ifdef TORCH_API_INCLUDE_EXTENSION_H

namespace {

/// helper classes for passing file-like object to SoxEffectChain
struct FileObjInputPriv {
  sox_format_t* sf;
  py::object* fileobj;
  char* buffer;
  uint64_t buffer_size;
};

/// Callback function to feed byte string
/// https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/sox.h#L1268-L1278
int fileobj_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  auto priv = static_cast<FileObjInputPriv *>(effp->priv);
  auto sf = priv->sf;
  auto fileobj = priv->fileobj;
  auto buffer = priv->buffer;
  auto buffer_size = priv->buffer_size;

  // 1. Refresh the buffer
  //
  // NOTE:
  //   Since the underlying FILE* was opened with `fmemopen`, the only way
  //   libsox detect EOF is reaching the end of the buffer. (null byte won't help)
  //   Therefore we need to align the content at the end of buffer, otherwise,
  //   libsox will keep reading the content beyond intended length.
  //
  // Before:
  //
  //     |<--------consumed------->|<-remaining->|
  //     |*************************|-------------|
  //                               ^ ftell
  //
  // After:
  //
  //     |<-offset->|<-remaining->|<--new data-->|
  //     |**********|-------------|++++++++++++++|
  //                ^ ftell

  const auto num_consumed = sf->tell_off;
  const auto num_remain = buffer_size - num_consumed;

  // 1.1. First, we fetch the data to see if there is data to fill the buffer
  py::bytes chunk_ = fileobj->attr("read")(num_consumed);
  const auto num_refill = py::len(chunk_);
  const auto offset = buffer_size - (num_remain + num_refill);

  if(num_refill > num_consumed) {
    std::ostringstream message;
    message << "Tried to read up to " << num_consumed << " bytes but, "
            << "received " << num_refill << " bytes. "
            << "The given object does not confirm to read protocol of file object.";
    throw std::runtime_error(message.str());
  }

  // 1.2. Move the unconsumed data towards the beginning of buffer.
  if (num_remain) {
    auto src = static_cast<void*>(buffer + num_consumed);
    auto dst = static_cast<void*>(buffer + offset);
    memmove(dst, src, num_remain);
  }

  // 1.3. Refill the remaining buffer.
  if (num_refill) {
    auto chunk = static_cast<std::string>(chunk_);
    auto src = static_cast<void*>(const_cast<char*>(chunk.c_str()));
    auto dst = buffer + offset + num_remain;
    memcpy(dst, src, num_refill);
  }

  // 1.4. Set the file pointer to the new offset
  sf->tell_off = offset;
  fseek ((FILE*)sf->fp, offset, SEEK_SET);

  // 2. Perform decoding operation
  // The following part is practically same as "input" effect
  // https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/input.c#L30-L48

  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % effp->out_signal.channels;

  // Read up to *osamp samples into obuf;
  // store the actual number read back to *osamp
  *osamp = sox_read(sf, obuf, *osamp);

  return *osamp? SOX_SUCCESS : SOX_EOF;
}

sox_effect_handler_t* get_fileobj_input_handler() {
  static sox_effect_handler_t handler{/*name=*/"input_fileobj_object",
                                      /*usage=*/NULL,
                                      /*flags=*/SOX_EFF_MCHAN,
                                      /*getopts=*/NULL,
                                      /*start=*/NULL,
                                      /*flow=*/NULL,
                                      /*drain=*/fileobj_input_drain,
                                      /*stop=*/NULL,
                                      /*kill=*/NULL,
                                      /*priv_size=*/sizeof(FileObjInputPriv)};
  return &handler;
}

} // namespace

void SoxEffectsChain::addInputFileObj(
    sox_format_t* sf,
    char* buffer,
    uint64_t buffer_size,
    py::object* fileobj) {
  in_sig_ = sf->signal;
  interm_sig_ = in_sig_;

  SoxEffect e(sox_create_effect(get_fileobj_input_handler()));
  auto priv = static_cast<FileObjInputPriv*>(e->priv);
  priv->sf = sf;
  priv->fileobj = fileobj;
  priv->buffer = buffer;
  priv->buffer_size = buffer_size;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error("Internal Error: Failed to add effect: input fileobj");
  }
}

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_effects_chain
} // namespace torchaudio
