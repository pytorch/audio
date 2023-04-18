import logging
import pathlib
from argparse import ArgumentParser

import ci_sdr

import torch
from datamodule import L3DAS22DataModule
from model import DNNBeamformer
from pesq import pesq
from pystoi import stoi

logger = logging.getLogger()


def run_eval(args):
    model = DNNBeamformer()
    checkpoint = torch.load(args.checkpoint_path)
    new_state_dict = {}
    for k in checkpoint["state_dict"].keys():
        if "loss" not in k:
            new_state_dict[k.replace("model.", "")] = checkpoint["state_dict"][k]
    model.load_state_dict(new_state_dict)
    model.eval()
    data_module = L3DAS22DataModule(dataset_path=args.dataset_path, batch_size=args.batch_size)
    if args.use_cuda:
        model = model.to(device="cuda")
    CI_SDR = 0.0
    STOI = 0.0
    PESQ = 0
    with torch.no_grad():
        for idx, batch in enumerate(data_module.test_dataloader()):
            mixture, clean, _, _ = batch
            if args.use_cuda:
                mixture = mixture.cuda()
            clean = clean[0]
            estimate = model(mixture).cpu()
            ci_sdr_v = (
                -ci_sdr.pt.ci_sdr(estimate, clean, compute_permutation=False, filter_length=512, change_sign=False)
                .mean()
                .item()
            )
            clean, estimate = clean[0].numpy(), estimate[0].numpy()
            stoi_v = stoi(clean, estimate, 16000, extended=False)
            pesq_v = pesq(16000, clean, estimate, "wb")
            CI_SDR += (1.0 / float(idx + 1)) * (ci_sdr_v - CI_SDR)
            STOI += (1.0 / float(idx + 1)) * (stoi_v - STOI)
            PESQ += (1.0 / float(idx + 1)) * (pesq_v - PESQ)
            if idx % 100 == 0:
                logger.warning(f"Processed elem {idx}; Ci-SDR: {CI_SDR}, stoi: {STOI}, pesq: {PESQ}")

        # visualize and save results
        results = {"Ci-SDR": CI_SDR, "stoi": STOI, "pesq": PESQ}
        print("*******************************")
        print("RESULTS")
        for i in results:
            print(i, results[i])


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        required=True,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to L3DAS22 datasets.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size for training. (Default: 4)",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    cli_main()
