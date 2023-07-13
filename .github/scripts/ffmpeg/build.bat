@echo off

set PROJ_FOLDER=%cd%

choco install -y --no-progress msys2 --package-parameters "/NoUpdate"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "pacman -S --noconfirm --needed base-devel mingw-w64-x86_64-toolchain diffutils"
C:\tools\msys64\usr\bin\env MSYSTEM=MINGW64 /bin/bash -l -c "cd ${PROJ_FOLDER} && packaging/vc_env_helper.bat bash .github/scripts/ffmpeg/build.sh"

:end
