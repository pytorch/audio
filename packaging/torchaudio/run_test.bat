@echo on

pip install --no-deps kaldi-io PySoundFile
if errorlevel 1 exit /b 1

pytest . --verbose --maxfail=1000000
if errorlevel 1 exit /b 1
