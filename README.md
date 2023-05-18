# Github Pages for Pytorch Audio

This branch holds the rendered html for Pytorch Audio. The pages are served via
github pages at https://pytorch.org/audio/, via a undocumented feature of
github: if one repo in the org has a [CNAME
file](https://github.com/pytorch/pytorch.github.io/blob/site/CNAME) (in this
case [pytorch/pytorch.github.io](https://github.com/pytorch/pytorch.github.io)),
then any other repo in that organization that turns on github pages will be
served under the same CNAME. The branch directory structure reflects the
release history of the project:
- each numbered directory holds the version of the documents at the time of
  release
- There are two special directories: `main` and `stable`.
  The first holds the
  current HEAD version of the documentation, and is updated each time `main`
  branch is updated.
  The `stable` directory is a symlink to the latest released version, and can
  be recreated using `ln -s` command.
- There is a simple top-level index.html that redirects to `stable/index.html`
  This is needed for naive links to https://pytorch.org/audio. Any search
  engine or external links should point to
  https://pytorch.org/audio/stable/index.html so they will not need the
  redirect.
