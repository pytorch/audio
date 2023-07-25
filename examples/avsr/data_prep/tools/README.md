## Face Recognition
We provide [ibug.face_detection](https://github.com/hhj1897/face_detection) in this repository. You can install directly from github repositories or by using compressed files.

### Option 1. Install from github repositories

* [Git LFS](https://git-lfs.github.com/), needed for downloading the pretrained weights that are larger than 100 MB.

You could install *`Homebrew`* and then install *`git-lfs`* without sudo priviledges.

```Shell
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```

### Option 2. Install by using compressed files

If you are experiencing over-quota issues for the above repositoies, you can download both packages [ibug.face_detection](https://www.doc.ic.ac.uk/~pm4115/tracker/face_detection.zip), unzip the files, and then run `pip install -e .` to install each package.

```Shell
wget https://www.doc.ic.ac.uk/~pm4115/tracker/face_detection.zip -O ./face_detection.zip
unzip -o ./face_detection.zip -d ./
cd face_detection
pip install -e .
cd ..
```
