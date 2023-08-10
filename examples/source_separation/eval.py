from argparse import ArgumentParser
from pathlib import Path

import mir_eval
import torch
from lightning_train import _get_dataloader, _get_model, sisdri_metric


def _eval(model, data_loader, device):
    results = torch.zeros(4)
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            mix, src, mask = batch
            mix, src, mask = mix.to(device), src.to(device), mask.to(device)
            est = model(mix)
            sisdri = sisdri_metric(est, src, mix, mask)
            src = src.cpu().detach().numpy()
            est = est.cpu().detach().numpy()
            mix = mix.repeat(1, src.shape[1], 1).cpu().detach().numpy()
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(src[0], est[0])
            sdr_mix, sir_mix, sar_mix, _ = mir_eval.separation.bss_eval_sources(src[0], mix[0])
            results += torch.tensor(
                [sdr.mean() - sdr_mix.mean(), sisdri, sir.mean() - sir_mix.mean(), sar.mean() - sar_mix.mean()]
            )
    results /= len(data_loader)
    print("SDR improvement: ", results[0].item())
    print("Si-SDR improvement: ", results[1].item())
    print("SIR improvement: ", results[2].item())
    print("SAR improvement: ", results[3].item())


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="librimix", type=str, choices=["wsj0mix", "librimix"])
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the directory where the directory ``Libri2Mix`` or ``Libri3Mix`` is stored.",
    )
    parser.add_argument(
        "--librimix-tr-split",
        default="train-360",
        choices=["train-360", "train-100"],
        help="The training partition of librimix dataset. (default: ``train-360``)",
    )
    parser.add_argument(
        "--librimix-task",
        default="sep_clean",
        type=str,
        choices=["sep_clean", "sep_noisy", "enh_single", "enh_both"],
        help="The task to perform (separation or enhancement, noisy or clean). (default: ``sep_clean``)",
    )
    parser.add_argument(
        "--num-speakers", default=2, type=int, help="The number of speakers in the mixture. (default: 2)"
    )
    parser.add_argument(
        "--sample-rate",
        default=8000,
        type=int,
        help="Sample rate of audio files in the given dataset. (default: 8000)",
    )
    parser.add_argument(
        "--exp-dir", default=Path("./exp"), type=Path, help="The directory to save checkpoints and logs."
    )
    parser.add_argument("--gpu-device", default=-1, type=int, help="The gpu device for model inference. (default: -1)")

    args = parser.parse_args()

    model = _get_model(num_sources=2)
    state_dict = torch.load(args.exp_dir / "best_model.pth")
    model.load_state_dict(state_dict)

    if args.gpu_device != -1:
        device = torch.device("cuda:" + str(args.gpu_device))
    else:
        device = torch.device("cpu")

    model = model.to(device)

    _, _, eval_loader = _get_dataloader(
        args.dataset,
        args.root_dir,
        args.num_speakers,
        args.sample_rate,
        1,  # batch size is set to 1 to avoid masking
        0,  # set num_workers to 0
        args.librimix_task,
        args.librimix_tr_split,
    )

    _eval(model, eval_loader, device)


if __name__ == "__main__":
    cli_main()
