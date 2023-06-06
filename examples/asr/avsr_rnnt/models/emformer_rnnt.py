from torchaudio.models.rnnt import emformer_rnnt_model


# https://pytorch.org/audio/master/_modules/torchaudio/models/rnnt.html#emformer_rnnt_base
def emformer_rnnt():
    return emformer_rnnt_model(
        input_dim=512,
        encoding_dim=1024,
        num_symbols=1024,
        segment_length=64,
        right_context_length=0,
        time_reduction_input_dim=128,
        time_reduction_stride=1,
        transformer_num_heads=4,
        transformer_ffn_dim=2048,
        transformer_num_layers=20,
        transformer_dropout=0.1,
        transformer_activation="gelu",
        transformer_left_context_length=30,
        transformer_max_memory_size=0,
        transformer_weight_init_scale_strategy="depthwise",
        transformer_tanh_on_mem=True,
        symbol_embedding_dim=512,
        num_lstm_layers=3,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
    )
