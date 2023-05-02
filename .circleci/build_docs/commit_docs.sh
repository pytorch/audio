#!/usr/bin/env bash

set -ex


if [ "$2" == "" ]; then
    echo call as "$0" "<src>" "<target branch>"
    echo where src is the root of the built documentation git checkout and
    echo branch should be "main" or "1.7" or so
    exit 1
fi

src=$1
target=$2

echo "committing docs from ${src} to ${target}"

pushd "${src}"
git checkout gh-pages
mkdir -p ./"${target}"
rm -rf ./"${target}"/*
cp -r "${src}/docs/build/html/"* ./"$target"
rm ./"$target"/artifact.tar.gz
git add --all ./"${target}" || true
git config user.email "soumith+bot@pytorch.org"
git config user.name "pytorchbot"
# If there aren't changes, don't make a commit; push is no-op
git commit -m "auto-generating sphinx docs" || true
git remote add https https://github.com/pytorch/audio.git
git push -u https gh-pages
