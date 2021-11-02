import json
import re
import sys
import argparse
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
    try:
        return subprocess.check_output(cmd).strip()
    except Exception:
        return None


def commit_title(commit_hash):
    cmd = ['git', 'log', '-n', '1', '--pretty=format:%s', f'{commit_hash}']
    return _run_cmd(cmd)


def parse_pr_number(commit_hash, title):
    regex = r"(#[0-9]+)"
    matches = re.findall(regex, title)
    if len(matches) == 0:
        print(f"[{commit_hash}: {title}] Could not parse PR number, ignoring PR")
        return None
    if len(matches) > 1:
        print(f"[{commit_hash}: {title}] Got two PR numbers, using the last one")
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
    request = requests.post("https://api.github.com/graphql", json={"query": query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


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
    edges = query["data"]["repository"]["pullRequest"]["labels"]["edges"]
    return [edge["node"]["name"] for edge in edges]


def get_features(commit_hash):
    title = commit_title(commit_hash)
    pr_number = parse_pr_number(commit_hash, title)
    labels = []
    if pr_number is not None:
        labels = gh_labels(pr_number)
    return Features(title, pr_number, labels)


def get_commits_between(base_version, new_version):
    cmd = ['git', 'merge-base', f'{base_version}', f'{new_version}']
    merge_base = _run_cmd(cmd)

    # Returns a list of items in the form
    # a7854f33 Add HuBERT model architectures (#1769)
    cmd = ['git', 'log', '--reverse', '--oneline', f'{merge_base}..{new_version}']
    commits = _run_cmd(cmd)

    log_lines = commits.split("\n")
    hashes, titles = zip(*[log_line.split(" ", 1) for log_line in log_lines])
    return hashes, titles


def _parse_args(args):
    parser = argparse.ArgumentParser()
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
    # Usage: python scripts/release_notes/retrieve_prs.py tags/v0.10.0 \
    # 18685a517ae68353b05b9a0ede5343df31525c76 --file data.json
    _main(_parse_args(sys.argv[1:]))
