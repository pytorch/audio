import logging
import torch
import torchaudio
import sentencepiece as spm
from argparse import ArgumentParser
from transforms import get_data_module


logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def get_lightning_module(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    if args.md == "av":
        from lightning_av import AVConformerRNNTModule
        model = AVConformerRNNTModule(args, sp_model)
    else:
        from lightning import ConformerRNNTModule
        model = ConformerRNNTModule(args, sp_model)
    ckpt = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def run_eval(model, data_module):
    total_edit_distance = 0
    total_length = 0
    dataloader = data_module.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            actual = sample[0][-1]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.warning(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.warning(f"Final WER: {total_edit_distance / total_length}")
    return total_edit_distance / total_length


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--md",
        type=str,
        help="Modality",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Perform online or offline recognition.",
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to LRW audio-visual datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to checkpoint model.",
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to Pretraned model.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to use debug level for logging"
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    data_module = get_data_module(args, str(args.sp_model_path))
    run_eval(model, data_module)


if __name__ == "__main__":
    cli_main()
