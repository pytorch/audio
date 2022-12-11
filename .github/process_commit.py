"""
This script finds the person responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'.
Note: we only ping the person who pulls the pr, not the reviewers, as the reviewers can sometimes be external
to torchaudio with no labeling responsibility, so we don't want to bother them.
"""

import json
import os
import sys
from typing import Any, Optional, Set, Tuple

import requests

# For a PR to be properly labeled it should have one primary label and one secondary label
# For a PR with primary label "other", it does not require an additional secondary label
PRIMARY_LABELS = {
    "BC-breaking",
    "deprecation",
    "bug fix",
    "new feature",
    "improvement",
    "tutorial",
    "recipe",
    "prototype",
    "other",
}

SECONDARY_LABELS = {
    "module: I/O",
    "module: ops",
    "module: models",
    "module: pipelines",
    "module: datasets",
    "module: docs",
    "module: tests",
    "build",
    "style",
    "perf",
    "other",
}
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REQUEST_HEADERS = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
TORCHAUDIO_REPO = "https://api.github.com/repos/pytorch/audio"


def query_torchaudio(cmd: str) -> Any:
    response = requests.get(f"{TORCHAUDIO_REPO}/{cmd}", headers=REQUEST_HEADERS)
    return response.json()


def get_pr_merger_and_number(commit_hash: str) -> Optional[str]:
    data = query_torchaudio(f"commits/{commit_hash}")
    commit_message = data["commit"]["message"]

    pulled_by = commit_message.split("Pulled By: ")
    pulled_by = pulled_by[1].split("\n")[0] if len(pulled_by) > 1 else None

    pr_number = commit_message.split("Pull Request resolved: https://github.com/pytorch/audio/pull/")
    pr_number = pr_number[1].split("\n")[0] if len(pr_number) > 1 else None

    return pulled_by, pr_number


def get_labels(pr_number: int) -> Set[str]:
    data = query_torchaudio(f"pulls/{pr_number}")
    labels = {label["name"] for label in data["labels"]}
    return labels


def post_github_comment(pr_number: int, merger: str) -> Any:
    message = {
        "body": f"Hey @{merger}."
        + """
You merged this PR, but labels were not properly added. Please add a primary and secondary label \
(See https://github.com/pytorch/audio/blob/main/.github/process_commit.py)"""
    }

    response = requests.post(
        f"{TORCHAUDIO_REPO}/issues/{pr_number}/comments", json.dumps(message), headers=REQUEST_HEADERS
    )
    return response.json()


if __name__ == "__main__":
    commit_hash = sys.argv[1]

    merger, pr_number = get_pr_merger_and_number(commit_hash)
    if pr_number:
        labels = get_labels(pr_number)
        is_properly_labeled = bool(PRIMARY_LABELS.intersection(labels) and SECONDARY_LABELS.intersection(labels))

        if not is_properly_labeled:
            post_github_comment(pr_number, merger)
