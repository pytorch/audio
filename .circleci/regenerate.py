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

import jinja2
from jinja2 import select_autoescape
import yaml
import os.path


PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9"]
CU_VERSIONS_DICT = {"linux": ["cpu", "cu102", "cu111","rocm4.1"],
                    "windows": ["cpu", "cu102", "cu111"],
                    "macos": ["cpu"]}


DOC_VERSION = ('linux', '3.8')


def build_workflows(prefix='', upload=False, filter_branch=None, indentation=6):
    w = []
    w += build_download_job(filter_branch)
    for btype in ["wheel", "conda"]:
        for os_type in ["linux", "macos", "windows"]:
            for python_version in PYTHON_VERSIONS:
                for cu_version in CU_VERSIONS_DICT[os_type]:
                    fb = filter_branch
                    if cu_version.startswith("rocm") and btype=="conda":
                        continue
                    if not fb and (os_type == 'linux' and
                                   btype == 'wheel' and
                                   python_version == '3.8' and
                                   cu_version == 'cpu'):
                        # the fields must match the build_docs "requires" dependency
                        fb = '/.*/'
                    w += build_workflow_pair(btype, os_type, python_version, cu_version, fb, prefix, upload)

    if not filter_branch:
        # Build on every pull request, but upload only on nightly and tags
        w += build_doc_job('/.*/')
        w += upload_doc_job('nightly')
        w += docstring_parameters_sync_job(None)


    return indent(indentation, w)


def build_download_job(filter_branch):
    job = {
        "name": "download_third_parties_nix",
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    return [{"download_third_parties_nix": job}]


def build_workflow_pair(btype, os_type, python_version, cu_version, filter_branch, prefix='', upload=False):

    w = []
    base_workflow_name = f"{prefix}binary_{os_type}_{btype}_py{python_version}_{cu_version}"
    w.append(generate_base_workflow(base_workflow_name, python_version, cu_version, filter_branch, os_type, btype))

    if upload:

        w.append(generate_upload_workflow(base_workflow_name, filter_branch, btype))

        if filter_branch == 'nightly' and os_type != 'macos':
            pydistro = 'pip' if btype == 'wheel' else 'conda'
            w.append(generate_smoketest_workflow(pydistro, base_workflow_name, filter_branch, python_version, cu_version, os_type))

    return w


def build_doc_job(filter_branch):
    job = {
        "name": "build_docs",
        "python_version": "3.8",
        "requires": ["binary_linux_wheel_py3.8_cpu", ],
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    return [{"build_docs": job}]


def upload_doc_job(filter_branch):
    job = {
        "name": "upload_docs",
        "context": "org-member",
        "python_version": "3.8",
        "requires": ["build_docs", ],
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    return [{"upload_docs": job}]


def docstring_parameters_sync_job(filter_branch):
    job = {
        "name": "docstring_parameters_sync",
        "python_version": "3.8",
        "requires": ["binary_linux_wheel_py3.8_cpu", ],
    }

    if filter_branch:
        job["filters"] = gen_filter_branch_tree(filter_branch)
    return [{"docstring_parameters_sync": job}]


def generate_base_workflow(base_workflow_name, python_version, cu_version, filter_branch, os_type, btype):

    d = {
        "name": base_workflow_name,
        "python_version": python_version,
        "cuda_version": cu_version,
    }

    if os_type in ['linux', 'macos']:
        d['requires'] = ['download_third_parties_nix']
    if btype == 'conda':
        d['conda_docker_image'] = f'pytorch/conda-builder:{cu_version.replace("cu1","cuda1")}'
    elif cu_version.startswith('cu'):
        d['wheel_docker_image'] = f'pytorch/manylinux-{cu_version.replace("cu1","cuda1")}'
    elif cu_version.startswith('rocm'):
        d["wheel_docker_image"] = f"pytorch/manylinux-rocm:{cu_version[len('rocm'):]}"

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {f"binary_{os_type}_{btype}": d}


def gen_filter_branch_tree(*branches):
    return {
        "branches": {
            "only": list(branches),
        },
        "tags": {
            # Using a raw string here to avoid having to escape
            # anything
            "only": r"/v[0-9]+(\.[0-9]+)*-rc[0-9]+/"
        }
    }


def generate_upload_workflow(base_workflow_name, filter_branch, btype):
    d = {
        "name": "{base_workflow_name}_upload".format(base_workflow_name=base_workflow_name),
        "context": "org-member",
        "requires": [base_workflow_name],
    }

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {"binary_{btype}_upload".format(btype=btype): d}


def generate_smoketest_workflow(pydistro, base_workflow_name, filter_branch, python_version, cu_version, os_type):

    required_build_suffix = "_upload"
    required_build_name = base_workflow_name + required_build_suffix

    smoke_suffix = f"smoke_test_{pydistro}".format(pydistro=pydistro)
    d = {
        "name": f"{base_workflow_name}_{smoke_suffix}",
        "requires": [required_build_name],
        "python_version": python_version,
        "cuda_version": cu_version,
    }

    if filter_branch:
        d["filters"] = gen_filter_branch_tree(filter_branch)

    return {f"smoke_test_{os_type}_{pydistro}": d}


def indent(indentation, data_list):
    return ("\n" + " " * indentation).join(yaml.dump(data_list).splitlines())


def unittest_workflows(indentation=6):
    jobs = []
    jobs += build_download_job(None)
    for os_type in ["linux", "windows", "macos"]:
        for device_type in ["cpu", "gpu"]:
            if os_type == "macos" and device_type == "gpu":
                continue

            for i, python_version in enumerate(PYTHON_VERSIONS):
                job = {
                    "name": f"unittest_{os_type}_{device_type}_py{python_version}",
                    "python_version": python_version,
                    "cuda_version": 'cpu' if device_type=="cpu" else "cu102",
                }

                if os_type != "windows":
                    job['requires'] = ['download_third_parties_nix']

                jobs.append({f"unittest_{os_type}_{device_type}": job})

                if i == 0 and os_type == "linux" and device_type == "cpu":
                    jobs.append({
                        "stylecheck": {
                            "name": f"stylecheck_py{python_version}",
                            "python_version": python_version,
                            "cuda_version": 'cpu' if device_type=="cpu" else "cu102",
                        }
                    })
    return indent(indentation, jobs)


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=select_autoescape(enabled_extensions=('html', 'xml')),
    )

    with open(os.path.join(d, 'config.yml'), 'w') as f:
        f.write(env.get_template('config.yml.in').render(
            build_workflows=build_workflows,
            unittest_workflows=unittest_workflows,
        ))
        f.write("\n")
