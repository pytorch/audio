#!/usr/bin/env python3

"""
This script should use a very simple, functional programming style.
Avoid Jinja macros in favor of native Python functions.

Don't go overboard on code generation; use Python only to generate
content that can't be easily declared statically using CircleCI's YAML API.

Data declarations (e.g. the nested loops for defining the configuration matrix)
should be at the top of the file for easy updating.

See this comment for design rationale:
https://github.com/pytorch/vision/pull/1321#issuecomment-531033978
"""

import os.path

import jinja2
import yaml
from jinja2 import select_autoescape


PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]


def build_download_job(filter_branch):
    job = {
        "name": "download_third_parties",
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    return [{"download_third_parties": job}]


def build_ffmpeg_job(os_type, filter_branch):
    job = {
        "name": f"build_ffmpeg_{os_type}",
        "requires": ["download_third_parties"],
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    job["python_version"] = "foo"
    return [{f"build_ffmpeg_{os_type}": job}]


def gen_filter_branch_tree(*branches):
    return {
        "branches": {
            "only": list(branches),
        },
        "tags": {
            # Using a raw string here to avoid having to escape
            # anything
            "only": r"/v[0-9]+(\.[0-9]+)*-rc[0-9]+/"
        },
    }


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(yaml.dump(data_list).splitlines())


def unittest_python_versions(os):
    return {
        "windows": PYTHON_VERSIONS[:1],
        "macos": PYTHON_VERSIONS[:1],
        "linux": PYTHON_VERSIONS,
    }.get(os)


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
    )

    with open(os.path.join(d, "config.yml"), "w") as f:
        f.write(
            env.get_template("config.yml.in").render()
        )
        f.write("\n")
