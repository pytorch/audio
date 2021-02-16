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
/// the way to pass extra data is to attach data to sox_effect_t::priv pointer.
/// The following structs will be assigned to sox_effect_t::priv pointer which
/// gives sox_effect_t an access to input Tensor and output buffer object.
struct TensorInputPriv {
  size_t index;
  torch::Tensor* waveform;
  int64_t sample_rate;
  bool channels_first;
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
  auto tensor = *(priv->waveform);
  auto num_channels = effp->out_signal.channels;

  // Adjust the number of samples to read
  const size_t num_samples = tensor.numel();
  if (index + *osamp > num_samples) {
    *osamp = num_samples - index;
  }
  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % num_channels;

  // Slice the input Tensor
  const auto tensor_ = [&]() {
    auto i_frame = index / num_channels;
    auto num_frames = *osamp / num_channels;
    auto t = (priv->channels_first)
        ? tensor.index({Slice(), Slice(i_frame, i_frame + num_frames)}).t()
        : tensor.index({Slice(i_frame, i_frame + num_frames), Slice()});
    return t.reshape({-1}).contiguous();
  }();

  // Convert to sox_sample_t (int32_t) and write to buffer
  SOX_SAMPLE_LOCALS;
  switch (tensor_.dtype().toScalarType()) {
    case c10::ScalarType::Float: {
      auto ptr = tensor_.data_ptr<float_t>();
      for (size_t i = 0; i < *osamp; ++i) {
        obuf[i] = SOX_FLOAT_32BIT_TO_SAMPLE(ptr[i], effp->clips);
      }
      break;
    }
    case c10::ScalarType::Int: {
      auto ptr = tensor_.data_ptr<int32_t>();
      for (size_t i = 0; i < *osamp; ++i) {
        obuf[i] = SOX_SIGNED_32BIT_TO_SAMPLE(ptr[i], effp->clips);
      }
      break;
    }
    case c10::ScalarType::Short: {
      auto ptr = tensor_.data_ptr<int16_t>();
      for (size_t i = 0; i < *osamp; ++i) {
        obuf[i] = SOX_SIGNED_16BIT_TO_SAMPLE(ptr[i], effp->clips);
      }
      break;
    }
    case c10::ScalarType::Byte: {
      auto ptr = tensor_.data_ptr<uint8_t>();
      for (size_t i = 0; i < *osamp; ++i) {
        obuf[i] = SOX_UNSIGNED_8BIT_TO_SAMPLE(ptr[i], effp->clips);
      }
      break;
    }
    default:
      throw std::runtime_error("Unexpected dtype.");
  }
  priv->index += *osamp;
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
  static sox_effect_handler_t handler{
      /*name=*/"input_tensor",
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
  static sox_effect_handler_t handler{
      /*name=*/"output_tensor",
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
  static sox_effect_handler_t handler{
      /*name=*/"output_file",
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

void SoxEffectsChain::addInputTensor(
    torch::Tensor* waveform,
    int64_t sample_rate,
    bool channels_first) {
  in_sig_ = get_signalinfo(waveform, sample_rate, "wav", channels_first);
  interm_sig_ = in_sig_;
  SoxEffect e(sox_create_effect(get_tensor_input_handler()));
  auto priv = static_cast<TensorInputPriv*>(e->priv);
  priv->index = 0;
  priv->waveform = waveform;
  priv->sample_rate = sample_rate;
  priv->channels_first = channels_first;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: input_tensor");
  }
}

void SoxEffectsChain::addOutputBuffer(
    std::vector<sox_sample_t>* output_buffer) {
  SoxEffect e(sox_create_effect(get_tensor_output_handler()));
  static_cast<TensorOutputPriv*>(e->priv)->buffer = output_buffer;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: output_tensor");
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

  auto returned_effect = sox_find_effect(name.c_str());
  if (!returned_effect) {
    std::ostringstream stream;
    stream << "Unsupported effect: " << name;
    throw std::runtime_error(stream.str());
  }
  SoxEffect e(sox_create_effect(returned_effect));
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
  bool eof_reached;
  char* buffer;
  uint64_t buffer_size;
};

struct FileObjOutputPriv {
  sox_format_t* sf;
  py::object* fileobj;
  char** buffer;
  size_t* buffer_size;
};

/// Callback function to feed byte string
/// https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/sox.h#L1268-L1278
int fileobj_input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  auto priv = static_cast<FileObjInputPriv*>(effp->priv);
  auto sf = priv->sf;
  auto buffer = priv->buffer;

  // 1. Refresh the buffer
  //
  // NOTE:
  //   Since the underlying FILE* was opened with `fmemopen`, the only way
  //   libsox detect EOF is reaching the end of the buffer. (null byte won't
  //   help) Therefore we need to align the content at the end of buffer,
  //   otherwise, libsox will keep reading the content beyond intended length.
  //
  // Before:
  //
  //     |<-------consumed------>|<---remaining--->|
  //     |***********************|-----------------|
  //                             ^ ftell
  //
  // After:
  //
  //     |<-offset->|<---remaining--->|<-new data->|
  //     |**********|-----------------|++++++++++++|
  //                ^ ftell

  // NOTE:
  //   Do not use `sf->tell_off` here. Presumably, `tell_off` and `fseek` are
  //   supposed to be in sync, but there are cases (Vorbis) they are not
  //   in sync and `tell_off` has seemingly uninitialized value, which
  //   leads num_remain to be negative and cause segmentation fault
  //   in `memmove`.
  const auto tell = ftell((FILE*)sf->fp);
  if (tell < 0) {
    throw std::runtime_error("Internal Error: ftell failed.");
  }
  const auto num_consumed = static_cast<size_t>(tell);
  if (num_consumed > priv->buffer_size) {
    throw std::runtime_error("Internal Error: buffer overrun.");
  }

  const auto num_remain = priv->buffer_size - num_consumed;

  // 1.1. Fetch the data to see if there is data to fill the buffer
  size_t num_refill = 0;
  std::string chunk(num_consumed, '\0');
  if (num_consumed && !priv->eof_reached) {
    num_refill = read_fileobj(
        priv->fileobj, num_consumed, const_cast<char*>(chunk.data()));
    if (num_refill < num_consumed) {
      priv->eof_reached = true;
    }
  }
  const auto offset = num_consumed - num_refill;

  // 1.2. Move the unconsumed data towards the beginning of buffer.
  if (num_remain) {
    auto src = static_cast<void*>(buffer + num_consumed);
    auto dst = static_cast<void*>(buffer + offset);
    memmove(dst, src, num_remain);
  }

  // 1.3. Refill the remaining buffer.
  if (num_refill) {
    auto src = static_cast<void*>(const_cast<char*>(chunk.c_str()));
    auto dst = buffer + offset + num_remain;
    memcpy(dst, src, num_refill);
  }

  // 1.4. Set the file pointer to the new offset
  sf->tell_off = offset;
  fseek((FILE*)sf->fp, offset, SEEK_SET);

  // 2. Perform decoding operation
  // The following part is practically same as "input" effect
  // https://github.com/dmkrepo/libsox/blob/b9dd1a86e71bbd62221904e3e59dfaa9e5e72046/src/input.c#L30-L48

  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % effp->out_signal.channels;

  // Read up to *osamp samples into obuf;
  // store the actual number read back to *osamp
  *osamp = sox_read(sf, obuf, *osamp);

  // Decoding is finished when fileobject is exhausted and sox can no longer
  // decode a sample.
  return (priv->eof_reached && !*osamp) ? SOX_EOF : SOX_SUCCESS;
}

int fileobj_output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  *osamp = 0;
  if (*isamp) {
    auto priv = static_cast<FileObjOutputPriv*>(effp->priv);
    auto sf = priv->sf;
    auto fp = static_cast<FILE*>(sf->fp);
    auto fileobj = priv->fileobj;
    auto buffer = priv->buffer;
    auto buffer_size = priv->buffer_size;

    // Encode chunk
    auto num_samples_written = sox_write(sf, ibuf, *isamp);
    fflush(fp);

    // Copy the encoded chunk to python object.
    fileobj->attr("write")(py::bytes(*buffer, ftell(fp)));

    // Reset FILE*
    sf->tell_off = 0;
    fseek(fp, 0, SEEK_SET);

    if (num_samples_written != *isamp) {
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

sox_effect_handler_t* get_fileobj_input_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"input_fileobj_object",
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

sox_effect_handler_t* get_fileobj_output_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"output_fileobj_object",
      /*usage=*/NULL,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/NULL,
      /*start=*/NULL,
      /*flow=*/fileobj_output_flow,
      /*drain=*/NULL,
      /*stop=*/NULL,
      /*kill=*/NULL,
      /*priv_size=*/sizeof(FileObjOutputPriv)};
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
  priv->eof_reached = false;
  priv->buffer = buffer;
  priv->buffer_size = buffer_size;
  if (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: input fileobj");
  }
}

void SoxEffectsChain::addOutputFileObj(
    sox_format_t* sf,
    char** buffer,
    size_t* buffer_size,
    py::object* fileobj) {
  out_sig_ = sf->signal;
  SoxEffect e(sox_create_effect(get_fileobj_output_handler()));
  auto priv = static_cast<FileObjOutputPriv*>(e->priv);
  priv->sf = sf;
  priv->fileobj = fileobj;
  priv->buffer = buffer;
  priv->buffer_size = buffer_size;
  if (sox_add_effect(sec_, e, &interm_sig_, &out_sig_) != SOX_SUCCESS) {
    throw std::runtime_error(
        "Internal Error: Failed to add effect: output fileobj");
  }
}

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_effects_chain
} // namespace torchaudio
