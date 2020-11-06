#!/usr/bin/env bash

set -ex


if [ "$1" == "" ]; then
    echo call as $0 "<target branch>", where branch should be "master" or "1.7" or so
    exit 1
fi

set -ex
echo "committing docs to ${target}"

git checkout gh-pages
rm -rf docs/$target/*
cp -r doc/build/html/* docs/$target
if [ $target == "master" ]; then
    rm -rf docs/_static/*
    cp -r doc/build/html/_static/* docs/_static
fi
git add docs || true
git config user.email "soumith+bot@pytorch.org"
git config user.name "pytorchbot"
# If there aren't changes, don't make a commit; push is no-op
git commit -m "auto-generating sphinx docs" || true
git push -u origin gh-pages


