from torchaudio.models import ConvTasNet


def conv_tasnet_base(num_sources: int = 2) -> ConvTasNet:
    r"""Builds the non-causal version of ConvTasNet in
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].

    The paramter settings follow the ones with the highest Si-SNR metirc score in the paper,
    except the mask activation function is changed from "sigmoid" to "relu" for performance improvement.

    Args:
        num_sources (int, optional): Number of sources in the output.
            (Default: 2)
    Returns:
        ConvTasNet:
            ConvTasNet model.
    """
    return ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=16,
        enc_num_feats=512,
        msk_kernel_size=3,
        msk_num_feats=128,
        msk_num_hidden_feats=512,
        msk_num_layers=8,
        msk_num_stacks=3,
        msk_activate="relu",
    )
