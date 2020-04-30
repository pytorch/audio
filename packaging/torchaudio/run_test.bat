@echo on

python -m pip install kaldi-io PySoundFile
if errorlevel 1 exit /b 1

pytest . --verbose --maxfail=1000000
if errorlevel 1 exit /b 1
