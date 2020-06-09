#include <torchaudio/csrc/sox_effect.h>
#include <torchaudio/csrc/sox_io.h>

namespace torch {
namespace audio {

int initialize_sox() {
  /* Initialization for sox effects.  Only initialize once  */
  return sox_init();
}

int shutdown_sox() {
  /* Shutdown for sox effects.  Do not shutdown between multiple calls  */
  return sox_quit();
}

std::vector<std::string> get_effect_names() {
  sox_effect_fn_t const* fns = sox_get_effect_fns();
  std::vector<std::string> sv;
  for (int i = 0; fns[i]; ++i) {
    const sox_effect_handler_t* eh = fns[i]();
    if (eh && eh->name)
      sv.push_back(eh->name);
  }
  return sv;
}

int build_flow_effects(
    const std::string& file_name,
    at::Tensor otensor,
    bool ch_first,
    sox_signalinfo_t* target_signal,
    sox_encodinginfo_t* target_encoding,
    const char* file_type,
    std::vector<SoxEffect> pyeffs,
    int max_num_eopts) {
  /* This function builds an effects flow and puts the results into a tensor.
     It can also be used to re-encode audio using any of the available encoding
     options in SoX including sample rate and channel re-encoding. */

  // open input
  sox_format_t* input =
      sox_open_read(file_name.c_str(), nullptr, nullptr, nullptr);
  if (input == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  // only used if target signal or encoding are null
  sox_signalinfo_t empty_signal;
  sox_encodinginfo_t empty_encoding;

  // set signalinfo and encodinginfo if blank
  if (target_signal == nullptr) {
    target_signal = &empty_signal;
    target_signal->rate = input->signal.rate;
    target_signal->channels = input->signal.channels;
    target_signal->length = SOX_UNSPEC;
    target_signal->precision = input->signal.precision;
#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
    target_signal->mult = nullptr;
#endif
  }
  if (target_encoding == nullptr) {
    target_encoding = &empty_encoding;
    target_encoding->encoding = SOX_ENCODING_SIGN2; // Sample format
    target_encoding->bits_per_sample =
        input->signal.precision; // Bits per sample
    target_encoding->compression = 0.0; // Compression factor
    target_encoding->reverse_bytes =
        sox_option_default; // Should bytes be reversed
    target_encoding->reverse_nibbles =
        sox_option_default; // Should nibbles be reversed
    target_encoding->reverse_bits =
        sox_option_default; // Should bits be reversed (pairs of bits?)
    target_encoding->opposite_endian = sox_false; // Reverse endianness
  }

  // check for rate or channels effect and change the output signalinfo
  // accordingly
  for (SoxEffect se : pyeffs) {
    if (se.ename == "rate") {
      target_signal->rate = std::stod(se.eopts[0]);
    } else if (se.ename == "channels") {
      target_signal->channels = std::stoi(se.eopts[0]);
    }
  }

  // create interm_signal for effects, intermediate steps change this in-place
  sox_signalinfo_t interm_signal = input->signal;

#ifdef __APPLE__
  // According to Mozilla Deepspeech sox_open_memstream_write doesn't work
  // with OSX
  char tmp_name[] = "/tmp/fileXXXXXX";
  int tmp_fd = mkstemp(tmp_name);
  close(tmp_fd);
  sox_format_t* output = sox_open_write(
      tmp_name, target_signal, target_encoding, "wav", nullptr, nullptr);
#else
  // create buffer and buffer_size for output in memwrite
  char* buffer;
  size_t buffer_size;
  // in-memory descriptor (this may not work for OSX)
  sox_format_t* output = sox_open_memstream_write(
      &buffer,
      &buffer_size,
      target_signal,
      target_encoding,
      file_type,
      nullptr);
#endif
  if (output == nullptr) {
    throw std::runtime_error("Error opening output memstream/temporary file");
  }
  // Setup the effects chain to decode/resample
  sox_effects_chain_t* chain =
      sox_create_effects_chain(&input->encoding, &output->encoding);

  sox_effect_t* e = sox_create_effect(sox_find_effect("input"));
  char* io_args[1];
  io_args[0] = (char*)input;
  sox_effect_options(e, 1, io_args);
  sox_add_effect(chain, e, &interm_signal, &input->signal);
  free(e);

  for (SoxEffect tae : pyeffs) {
    if (tae.ename == "no_effects")
      break;
    e = sox_create_effect(sox_find_effect(tae.ename.c_str()));
    e->global_info->global_info->verbosity = 1;
    if (tae.eopts[0] == "") {
      sox_effect_options(e, 0, nullptr);
    } else {
      int num_opts = tae.eopts.size();
      char* sox_args[max_num_eopts];
      for (std::vector<std::string>::size_type i = 0; i != tae.eopts.size();
           i++) {
        sox_args[i] = (char*)tae.eopts[i].c_str();
      }
      if (sox_effect_options(e, num_opts, sox_args) != SOX_SUCCESS) {
#ifdef __APPLE__
        unlink(tmp_name);
#endif
        throw std::runtime_error(
            "invalid effect options, see SoX docs for details");
      }
    }
    sox_add_effect(chain, e, &interm_signal, &output->signal);
    free(e);
  }

  e = sox_create_effect(sox_find_effect("output"));
  io_args[0] = (char*)output;
  sox_effect_options(e, 1, io_args);
  sox_add_effect(chain, e, &interm_signal, &output->signal);
  free(e);

  // Finally run the effects chain
  sox_flow_effects(chain, nullptr, nullptr);
  sox_delete_effects_chain(chain);

  // Close sox handles, buffer does not get properly sized until these are
  // closed
  sox_close(output);
  sox_close(input);

  int sr;
  // Read the in-memory audio buffer or temp file that we just wrote.
#ifdef __APPLE__
  /*
     Temporary filetype must have a valid header.  Wav seems to work here while
     raw does not.  Certain effects like chorus caused strange behavior on the
     mac.
  */
  // read_audio_file reads the temporary file and returns the sr and otensor
  sr = read_audio_file(
      tmp_name, otensor, ch_first, 0, 0, target_signal, target_encoding, "wav");
  // delete temporary audio file
  unlink(tmp_name);
#else
  // Resize output tensor to desired dimensions, different effects result in
  // output->signal.length, interm_signal.length and buffer size being
  // inconsistent with the result of the file output. We prioritize in the
  // order: output->signal.length > interm_signal.length > buffer_size Could be
  // related to: https://sourceforge.net/p/sox/bugs/314/
  int nc, ns;
  if (output->signal.length == 0) {
    // sometimes interm_signal length is extremely large, but the buffer_size
    // is double the length of the output signal
    if (interm_signal.length > (buffer_size * 10)) {
      ns = buffer_size / 2;
    } else {
      ns = interm_signal.length;
    }
    nc = interm_signal.channels;
  } else {
    nc = output->signal.channels;
    ns = output->signal.length;
  }
  otensor.resize_({ns / nc, nc});
  otensor = otensor.contiguous();

  input = sox_open_mem_read(
      buffer, buffer_size, target_signal, target_encoding, file_type);
  std::vector<sox_sample_t> samples(buffer_size);
  const int64_t samples_read = sox_read(input, samples.data(), buffer_size);
  assert(samples_read != nc * ns && samples_read != 0);
  AT_DISPATCH_ALL_TYPES(otensor.scalar_type(), "effects_buffer", [&] {
    auto* data = otensor.data_ptr<scalar_t>();
    std::copy(samples.begin(), samples.begin() + samples_read, data);
  });
  // free buffer and close mem_read
  sox_close(input);
  free(buffer);

  if (ch_first) {
    otensor.transpose_(1, 0);
  }
  sr = target_signal->rate;

#endif
  // return sample rate, output tensor modified in-place
  return sr;
}

} // namespace audio
} // namespace torch
