from torchaudio.prototype.models import conformer_rnnt_model

# https://pytorch.org/audio/master/_modules/torchaudio/prototype/models/rnnt.html#conformer_rnnt_model


def conformer_rnnt():
    return conformer_rnnt_model(
        input_dim=512,
        encoding_dim=1024,
        time_reduction_stride=1,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )
