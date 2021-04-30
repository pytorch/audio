#include <torch/script.h>

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
    m.def("rnnt_loss(Tensor logits,"
                    "Tensor targets,"
                    "Tensor src_lengths,"
                    "Tensor tgt_lengths,"
                    "int blank,"
                    "float clamp,"
                    "Tensor? wp_ends=None,"
                    "int l_buffer=0,"
                    "int r_buffer=0,"
                    "bool fused_log_smax=True,"
                    "bool reuse_logits_for_grads=True) -> (Tensor, Tensor?)");
}
