import os
import math
import glob
import ffmpeg
import shutil
from data.data_module import AVSRDataLoader
from utils import split_file, save_vid_aud_txt

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser


def load_args(default_config=None):
    parser = ArgumentParser()
    # -- for benchmark evaluation
    parser.add_argument("--data-dir",
        type=str,
        help="The directory for sequence.",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        help="The directory of saved mouth patches or embeddings.",
    )
    parser.add_argument(
        '--job-index',
        type=int,
        default=0,
        help='job index'
    )
    parser.add_argument(
        '--groups',
        type=int,
        default=1,
        help='specify the number of threads to be used',
    )
    parser.add_argument(
        '--folder',
        type=str,
        default="test",
        help='specify the set used in the experiment',
    )
    args = parser.parse_args()
    return args
args = load_args()

seg_duration = 16
dataset_name = "LRS3"
detector = "retinaface"

args.data_dir = os.path.normpath(args.data_dir)
vid_dataloader = AVSRDataLoader(modality="video", detector=detector, resize=(96, 96))
aud_dataloader = AVSRDataLoader(modality="audio")
# Step 2, extract mouth patches from segments.
seg_vid_len = seg_duration * 25
seg_aud_len = seg_duration * 16000

label_filename = os.path.join(
    args.dst_dir,
    dataset_name,
    f"{args.folder}_transcript_lengths_seg{seg_duration}s.csv" if args.groups <= 1 else \
    f"{args.folder}_transcript_lengths_seg{seg_duration}s.{args.groups}.{args.job_index}.csv"
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")
# Step 2, extract mouth patches from segments.
dst_vid_dir = os.path.join(args.dst_dir, dataset_name, dataset_name+f"_video_seg{seg_duration}s")
dst_txt_dir = os.path.join(args.dst_dir, dataset_name, dataset_name+f"_text_seg{seg_duration}s")
if args.folder == "test":
    filenames = glob.glob(os.path.join(args.data_dir, args.folder, "**", "*.mp4"), recursive=True)
elif args.folder == "train":
    filenames = glob.glob(os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True)
    filenames.extend(glob.glob(os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True))
    filenames.sort()
else:
    raise NotImplementedError

unit = math.ceil(len(filenames) * 1. / args.groups)
filenames = filenames[args.job_index * unit : (args.job_index+1) * unit]

for data_filename in filenames:
    try:
        video_data = vid_dataloader.load_data(data_filename)
        audio_data = aud_dataloader.load_data(data_filename)
    except UnboundLocalError:
        continue
    
    if  os.path.normpath(data_filename).split(os.sep)[-3] in ["trainval", "test", "main"]:
        dst_vid_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
        dst_aud_filename = f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.wav"
        dst_txt_filename = f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"
        trim_vid_data, trim_aud_data = video_data, audio_data
        text_line_list = open(data_filename[:-4]+".txt", 'r').read().splitlines()[0].split(' ')
        text_line = ' '.join(text_line_list[2:])
        content = text_line.replace("}","").replace("{","")

        if trim_vid_data is None or trim_aud_data is None:
            continue
        video_length = len(trim_vid_data)
        audio_length = trim_aud_data.size(1)
        if video_length == 0 or audio_length == 0:
            continue
        if audio_length/video_length < 560. or audio_length/video_length > 720. or video_length < 12:
            continue
        save_vid_aud_txt(dst_vid_filename, dst_aud_filename, dst_txt_filename,
            trim_vid_data, trim_aud_data, content, video_fps=25, audio_sample_rate=16000)

        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(in1['v'], in2['a'], dst_vid_filename[:-4]+'.m.mp4', vcodec='copy', acodec='aac', strict='experimental', loglevel="panic")
        out.run()
        os.remove(dst_aud_filename)
        os.remove(dst_vid_filename)
        shutil.move(dst_vid_filename[:-4]+'.m.mp4', dst_vid_filename)

        basename = dst_vid_filename.replace(dst_vid_dir+"/", "")
        f.write("{}\n".format(f"{dataset_name.lower()},{basename},{trim_vid_data.shape[0]},{len(content)}"))
        continue

    splitted = split_file(data_filename[:-4]+".txt", max_frames=seg_vid_len)
    for i in range(len(splitted)):
        if len(splitted) == 1:
            content, start, end, duration = splitted[i]
            trim_vid_data, trim_aud_data = video_data, audio_data
        else:
            content, start, end, duration = splitted[i]
            start_idx, end_idx = int(start * 25), int(end * 25)
            try:
                trim_vid_data, trim_aud_data = video_data[start_idx:end_idx], audio_data[:, start_idx*640:end_idx*640]
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
        if audio_length/video_length < 560. or audio_length/video_length > 720. or video_length < 12:
            continue
        save_vid_aud_txt(dst_vid_filename, dst_aud_filename, dst_txt_filename,
            trim_vid_data, trim_aud_data, content, video_fps=25, audio_sample_rate=16000)

        in1 = ffmpeg.input(dst_vid_filename)
        in2 = ffmpeg.input(dst_aud_filename)
        out = ffmpeg.output(in1['v'], in2['a'], dst_vid_filename[:-4]+'.m.mp4', vcodec='copy', acodec='aac', strict='experimental', loglevel="panic")
        out.run()
        os.remove(dst_aud_filename)
        os.remove(dst_vid_filename)
        shutil.move(dst_vid_filename[:-4]+'.m.mp4', dst_vid_filename)

        basename = dst_vid_filename.replace(dst_vid_dir+"/", "")
        f.write("{}\n".format(f"{dataset_name.lower()},{basename},{trim_vid_data.shape[0]},{len(content)}"))
f.close()
