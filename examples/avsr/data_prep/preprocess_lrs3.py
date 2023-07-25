import argparse
import glob
import math
import os
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="LRS3 Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    help="The directory for sequence.",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Face detector used in the experiment.",
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Specify the dataset name used in the experiment",
)
parser.add_argument(
    "--root-dir",
    type=str,
    help="The root directory of cropped-face dataset.",
)
parser.add_argument(
    "--subset",
    type=str,
    required=True,
    help="Subset of the dataset used in the experiment.",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Length of the segment in seconds.",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing).",
)
args = parser.parse_args()

seg_duration = args.seg_duration
dataset = args.dataset

args.data_dir = os.path.normpath(args.data_dir)
vid_dataloader = AVSRDataLoader(modality="video", detector=args.detector, resize=(96, 96))
aud_dataloader = AVSRDataLoader(modality="audio")
# Step 2, extract mouth patches from segments.
seg_vid_len = seg_duration * 25
seg_aud_len = seg_duration * 16000

label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.csv"
    if args.groups <= 1
    else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.{args.groups}.{args.job_index}.csv",
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")
# Step 2, extract mouth patches from segments.
dst_vid_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_video_seg{seg_duration}s"
)
dst_txt_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_text_seg{seg_duration}s"
)
if args.subset == "test":
    filenames = glob.glob(os.path.join(args.data_dir, args.subset, "**", "*.mp4"), recursive=True)
elif args.subset == "train":
    filenames = glob.glob(os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True)
    filenames.extend(glob.glob(os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True))
    filenames.sort()
else:
    raise NotImplementedError

unit = math.ceil(len(filenames) * 1.0 / args.groups)
filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]

for data_filename in tqdm(filenames):
    try:
        video_data = vid_dataloader.load_data(data_filename)
        audio_data = aud_dataloader.load_data(data_filename)
    except UnboundLocalError:
        continue

    if os.path.normpath(data_filename).split(os.sep)[-3] in ["trainval", "test"]:
        dst_vid_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
        dst_aud_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.wav"
        dst_txt_filename = f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"
        trim_vid_data, trim_aud_data = video_data, audio_data
        text_line_list = open(data_filename[:-4] + ".txt", "r").read().splitlines()[0].split(" ")
        text_line = " ".join(text_line_list[2:])
        content = text_line.replace("}", "").replace("{", "")

        if trim_vid_data is None or trim_aud_data is None:
            continue
        video_length = len(trim_vid_data)
        audio_length = trim_aud_data.size(1)
        if video_length == 0 or audio_length == 0:
            continue
        if audio_length / video_length < 560.0 or audio_length / video_length > 720.0 or video_length < 12:
            continue
        save_vid_aud_txt(
            dst_vid_filename,
            dst_aud_filename,
            dst_txt_filename,
            trim_vid_data,
            trim_aud_data,
            content,
            video_fps=25,
            audio_sample_rate=16000,
        )

        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(
            in1["v"],
            in2["a"],
            dst_vid_filename[:-4] + ".m.mp4",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        out.run()
        os.remove(dst_aud_filename)
        os.remove(dst_vid_filename)
        shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)

        basename = os.path.relpath(dst_vid_filename, start=os.path.join(args.root_dir, dataset))
        f.write("{}\n".format(f"{dataset},{basename},{trim_vid_data.shape[0]},{len(content)}"))
        continue

    splitted = split_file(data_filename[:-4] + ".txt", max_frames=seg_vid_len)
    for i in range(len(splitted)):
        if len(splitted) == 1:
            content, start, end, duration = splitted[i]
            trim_vid_data, trim_aud_data = video_data, audio_data
        else:
            content, start, end, duration = splitted[i]
            start_idx, end_idx = int(start * 25), int(end * 25)
            try:
                trim_vid_data, trim_aud_data = (
                    video_data[start_idx:end_idx],
                    audio_data[:, start_idx * 640 : end_idx * 640],
                )
            except TypeError:
                continue
        dst_vid_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        dst_aud_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        dst_txt_filename = f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}_{i:02d}.txt"

        if trim_vid_data is None or trim_aud_data is None:
            continue
        video_length = len(trim_vid_data)
        audio_length = trim_aud_data.size(1)
        if video_length == 0 or audio_length == 0:
            continue
        if audio_length / video_length < 560.0 or audio_length / video_length > 720.0 or video_length < 12:
            continue
        save_vid_aud_txt(
            dst_vid_filename,
            dst_aud_filename,
            dst_txt_filename,
            trim_vid_data,
            trim_aud_data,
            content,
            video_fps=25,
            audio_sample_rate=16000,
        )

        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(
            in1["v"],
            in2["a"],
            dst_vid_filename[:-4] + ".m.mp4",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        out.run()
        os.remove(dst_aud_filename)
        os.remove(dst_vid_filename)
        shutil.move(dst_vid_filename[:-4] + ".m.mp4", dst_vid_filename)

        basename = os.path.relpath(dst_vid_filename, start=os.path.join(args.root_dir, dataset))
        f.write("{}\n".format(f"{dataset},{basename},{trim_vid_data.shape[0]},{len(content)}"))
f.close()
