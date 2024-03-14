"""Collect the PRs between two specified tags or commits and
    output the commit titles, PR numbers, and labels in a json file.
Usage: python tools/release_notes/retrieve_prs.py tags/v0.10.0 \
    18685a517ae68353b05b9a0ede5343df31525c76 --file data.json
"""

import argparse
import json
import re
import subprocess
from collections import namedtuple
from os.path import expanduser

import requests


Features = namedtuple(
    "Features",
    [
        "title",
        "pr_number",
        "labels",
    ],
)


def _run_cmd(cmd):
    return subprocess.check_output(cmd).decode("utf-8").strip()


def commit_title(commit_hash):
    cmd = ["git", "log", "-n", "1", "--pretty=format:%s", f"{commit_hash}"]
    return _run_cmd(cmd)


def parse_pr_number(title):
    regex = r"(#[0-9]+)"
    matches = re.findall(regex, title)
    if len(matches) == 0:
        print(f"[{title}] Could not parse PR number, ignoring PR")
        return None
    if len(matches) > 1:
        print(f"[{title}] Got two PR numbers, using the last one")
        return matches[-1][1:]
    return matches[0][1:]


def get_ghstack_token():
    pattern = "github_oauth = (.*)"
    with open(expanduser("~/.ghstackrc"), "r+") as f:
        config = f.read()
    matches = re.findall(pattern, config)
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    return matches[0]


token = get_ghstack_token()
headers = {"Authorization": f"token {token}"}


def run_query(query):
    response = requests.post("https://api.github.com/graphql", json={"query": query}, headers=headers)
    response.raise_for_status()
    return response.json()


def gh_labels(pr_number):
    query = f"""
    {{
      repository(owner: "pytorch", name: "audio") {{
        pullRequest(number: {pr_number}) {{
          labels(first: 10) {{
            edges {{
              node {{
                name
              }}
            }}
          }}
        }}
      }}
    }}
    """
    query = run_query(query)
    pr = query["data"]["repository"]["pullRequest"]
    if not pr:
        # to account for unrecognized PR numbers from commits originating from fb internal
        return []
    edges = pr["labels"]["edges"]
    return [edge["node"]["name"] for edge in edges]


def get_features(commit_hash):
    title = commit_title(commit_hash)
    pr_number = parse_pr_number(title)
    labels = []
    if pr_number is not None:
        labels = gh_labels(pr_number)
    return Features(title, pr_number, labels)


def get_merge_base(base_version, new_version):
    cmd = ["git", "merge-base", f"{base_version}", f"{new_version}"]
    merge_base = _run_cmd(cmd)
    return merge_base


def get_commits_between(base_version, new_version):
    merge_base = get_merge_base(base_version, new_version)

    # Returns a list of items in the form
    # a7854f33 Add HuBERT model architectures (#1769)
    cmd = ["git", "log", "--reverse", "--oneline", f"{merge_base}..{base_version}"]
    base_commits = _run_cmd(cmd).split("\n")
    base_prs = [parse_pr_number(commit) for commit in base_commits]

    cmd = ["git", "log", "--reverse", "--oneline", f"{merge_base}..{new_version}"]
    new_commits = _run_cmd(cmd).split("\n")

    commits = [commit for commit in new_commits if parse_pr_number(commit) not in base_prs]
    hashes, titles = zip(*[commit.split(" ", 1) for commit in commits])
    return hashes, titles


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("base_version", type=str, help="starting tag or commit (exclusive)")
    parser.add_argument("new_version", type=str, help="final tag or commit (inclusive)")
    parser.add_argument("--file", type=str, default="data.json", help="output json file")
    return parser.parse_args(args)


def _main(args):
    hashes, titles = get_commits_between(args.base_version, args.new_version)
    data = {}

    for idx, commit in enumerate(hashes):
        data[commit] = get_features(commit)
        if idx % 10 == 0:
            print(f"{idx} / {len(hashes)}")

    data = {commit: features._asdict() for commit, features in data.items()}
    with open(args.file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    _main(_parse_args())
