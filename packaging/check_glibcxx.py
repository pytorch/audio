"""
The goal of this script is to ensure that the .so files we ship do not contain
symbol versions from libstdc++ that are too recent. This is a very manual way of
doing the checks that `auditwheel repair` would do (but using auditwheel isn't
necessarily easy either).

Why this is needed: during development, we observed the following general
scenario in various local development setups:
- torchcodec is compiled with a given (somewhat recent) c++ toolchain (say
  gcc11)
- because the toolchain is recent, some recent symbol versions from libstdc++
  are added as dependencies in the torchcodec?.so files, e.g. GLIBCXX_3.4.29
  (this is normal)
- at runtime, for whatever reason, the libstdc++.so that gets loaded is *not*
  the one that was used when building. The libstdc++.so that is loaded can be
  older than the toolchain one, and it doesn't contain the more recent symbols
  that torchcodec?.so depends on, which leads to a runtime error.

The reasons why a different libstdc++.so is loaded at runtime can be multiple
(and mysterious! https://hackmd.io/@_NznxihTSmC-IgW4cgnlyQ/HJXc4BEHR).

This script doesn't try to prevent *that* (it's impossible anyway, as we don't
control users' environments). Instead, it prevents the dependency of torchcodec
on recent symbol versions, which ensures that torchcodec can run on both recent
*and* older runtimes.
The most recent symbol on the manylinux torch.2.3.1 wheel is
GLIBCXX_3.4.19, so as long as torchcodec doesn't ship a symbol that is higher
than that, torchcodec should be fine.

The easiest way to avoid recent symbols is simply to use an old-enough
toolchain. Relying on the test-infra runners should be enough.
"""

import re
import sys

if len(sys.argv) != 2:
    raise ValueError("Wrong usage: python check_glibcxx.py <str_with_symbols>.")

MAX_ALLOWED = (3, 4, 24)

symbol_matches = sys.argv[1].split("\n")
all_symbols = set()
for line in symbol_matches:
    # We search for GLIBCXX_major.minor.micro
    if match := re.search(r"GLIBCXX_\d+\.\d+\.\d+", line):
        all_symbols.add(match.group(0))

if not all_symbols:
    raise ValueError(
        f"No GLIBCXX symbols found in {symbol_matches}. Something is wrong."
    )

all_versions = (symbol.split("_")[1].split(".") for symbol in all_symbols)
all_versions = (tuple(int(v) for v in version) for version in all_versions)
max_version = max(all_versions)

print(f"Found the following GLIBCXX symbol versions: {all_symbols}.")
print(f"The max version is {max_version}. Max allowed is {MAX_ALLOWED}.")

if max_version > MAX_ALLOWED:
    raise AssertionError(
        "The max version is greater than the max allowed! "
        "That may leads to compatibility issues. "
        "Was the wheel compiled with an old-enough toolchain?"
    )

print("All good.")
