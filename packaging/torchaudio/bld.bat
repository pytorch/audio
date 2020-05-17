@echo off

echo %CD%
sh packaging/download_deps.sh "$(pwd)"
if errorlevel 1 exit /b 1

set IS_CONDA=1

python setup.py install --single-version-externally-managed --record=record.txt
