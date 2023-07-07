import os
from argparse import ArgumentParser


def load_args(default_config=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specify the dataset name used in the experiment",
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="Specify the set used in the experiment",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="The root directory of saved mouth patches or embeddings.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        help="Specify the number of threads to be used",
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=16,
        help="Specify the segment length",
    )
    args = parser.parse_args()
    return args


args = load_args()

dataset = args.dataset
subset = args.subset
seg_duration = args.seg_duration

# Check that there is more than one group
assert args.groups > 1, "There is no need to use this script for merging when --groups is 1."

# Create the filename template for label files
label_template = os.path.join(
    args.root_dir, "labels", f"{dataset}_{subset}_transcript_lengths_seg{seg_duration}s.{args.groups}"
)

lines = []
for job_index in range(args.groups):
    label_filename = f"{label_template}.{job_index}.csv"
    assert os.path.exists(label_filename), f"{label_filename} does not exist."

    with open(label_filename, "r") as file:
        lines.extend(file.read().splitlines())

# Write the merged labels to a new file
dst_label_filename = os.path.join(
    args.root_dir, dataset, f"{dataset}_{subset}_transcript_lengths_seg{seg_duration}s.csv"
)

with open(dst_label_filename, "w") as file:
    file.write("\n".join(lines))

# Print the number of files and total duration in hours
total_duration = sum(int(line.split(",")[2]) for line in lines) / 3600.0 / 25.0
print(f"The completed set has {len(lines)} files with a total of {total_duration} hours.")

# Remove the label files for each job index
print("** Remove the temporary label files **")
for job_index in range(args.groups):
    label_filename = f"{label_template}.{job_index}.csv"
    if os.path.exists(label_filename):
        os.remove(label_filename)

print("** Finish **")
