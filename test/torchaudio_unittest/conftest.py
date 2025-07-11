import pytest
import csv

def pytest_collection_modifyitems(config, items):
    with open('ffmpeg_fail_ids.txt', 'r') as file:
        fail_ids = set([f.strip() for f in file.readlines()])

    skip_marker = pytest.mark.skip(reason="FFMPEG incompatible with CI runner")

    for item in items:
        if item.nodeid in fail_ids:
            item.add_marker(skip_marker)
