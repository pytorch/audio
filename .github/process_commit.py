"""
This script finds the person responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'.
Note: we only ping the person who pulls the pr, not the reviewers, as the reviewers can sometimes be external
to torchaudio with no labeling responsibility, so we don't want to bother them.
"""

import json
import os
import sys
from typing import Any, Optional, Set

import requests

# For a PR to be properly labeled it should have one primary label and one secondary label
# For a PR with primary label "other", it does not require an additional secondary label
PRIMARY_LABELS = {
    "BC-breaking",
    "deprecation",
    "bug fix",
    "new feature",
    "improvement",
    "prototype",
    "other",
}

SECONDARY_LABELS = {
    "module: io",
    "module: ops",
    "module: models",
    "module: pipelines",
    "module: datasets",
    "module: docs",
    "module: tests",
    "tutorial",
    "recipe",
    "example",
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
(See https://github.com/pytorch/audio/blob/main/.github/process_commit.py).

---

## Some guidance:

Use 'module: ops' for operations under 'torchaudio/{transforms, functional}', \
and ML-related components under 'torchaudio/csrc' (e.g. RNN-T loss).

Things in "examples" directory:
- 'recipe' is applicable to training recipes under the 'examples' folder,
- 'tutorial' is applicable to tutorials under the “examples/tutorials” folder
- 'example' is applicable to everything else (e.g. C++ examples)
- 'module: docs' is applicable to code documentations (not to tutorials).

Regarding examples in code documentations, please also use 'module: docs'.

Please use 'other' tag only when you’re sure the changes are not much relevant to users, \
or when all other tags are not applicable. Try not to use it often, in order to minimize \
efforts required when we prepare release notes.

---

When preparing release notes, please make sure 'documentation' and 'tutorials' occur as the \
last sub-categories under each primary category like 'new feature', 'improvements' or 'prototype'.

Things related to build are by default excluded from the release note, \
except when it impacts users. For example:
    * Drop support of Python 3.7.
    * Add support of Python 3.X.
    * Change the way a third party library is bound (so that user needs to install it separately).
"""
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
