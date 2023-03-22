@echo off

if exist "third_party\ffmpeg\" goto end
if defined CIRCLECI (set PROJ_FOLDER="/c/Users/circleci/project") else (set PROJ_FOLDER=%cd%)

choco install -y --no-progress msys2 --package-parameters "/NoUpdate"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "pacman -S --noconfirm --needed base-devel mingw-w64-x86_64-toolchain diffutils"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "cd ${PROJ_FOLDER} && export FFMPEG_ROOT=${PWD}/third_party/ffmpeg && packaging/vc_env_helper.bat bash ./packaging/ffmpeg/build.sh"

:end
