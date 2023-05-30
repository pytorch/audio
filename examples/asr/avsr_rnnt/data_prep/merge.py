import os
from argparse import ArgumentParser


def load_args(default_config=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--dst-dir",
        type=str,
        default=None,
        help="The directory of saved mouth patches or embeddings.",
    )
    parser.add_argument(
        '--groups',
        type=int,
        default=1,
        help='Specify the number of threads to be used',
    )
    parser.add_argument(
        '--folder',
        type=str,
        default="test",
        help='Specify the set used in the experiment',
    )
    args = parser.parse_args()
    return args
args = load_args()

dataset_name = "LRS3"
seg_duration = 16

# Check that there is more than one group
assert args.groups > 1, "There is no need to use this script for merging when --groups is 1."

# Collect label filenames from each job index
lines = []
for job_index in range(args.groups):
    label_filename = os.path.join(
        args.dst_dir,
        dataset_name,
        f"{args.folder}_transcript_lengths_seg{seg_duration}s.{args.groups}.{job_index}.csv"
    )
    assert os.path.exists(label_filename), f"{label_filename} does not exist."
    lines.extend(open(label_filename).read().splitlines())

# Write the merged labels to a new file
dst_label_filename = os.path.join(
    args.dst_dir,
    dataset_name,
    f"{args.folder}_transcript_lengths_seg{seg_duration}s.csv"
)
with open(dst_label_filename, "w") as f:
    for line in lines:
        f.write(f"{line}\n")

# Print the number of files and total duration in hours
total_duration = sum([int(_.split(',')[2]) for _ in lines])/3600./25.:.2f
print(f"The completed set has {len(lines)} files with a total of {total_duration} hours.")

# Remove the label files for each job index
for job_index in range(args.groups):
    label_filename = os.path.join(
        args.dst_dir,
        dataset_name,
        f"{args.folder}_transcript_lengths_seg{seg_duration}s.{args.groups}.{job_index}.csv"
    )
    print(f"rm -rf {label_filename}")
    os.system(f"rm -rf {label_filename}")
