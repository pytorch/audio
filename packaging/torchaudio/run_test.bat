@echo on

pip install --user kaldi-io PySoundFile
if errorlevel 1 exit /b 1

dir %APPDATA%\Python\Python36\site-packages
echo %PYTHONPATH%

if "%PYTHONPATH%" == "" (
    set PYTHONPATH=%APPDATA%\Python\Python36\site-packages
) else (
    set PYTHONPATH=%PYTHONPATH%;%APPDATA%\Python\Python36\site-packages
)

echo %PYTHONPATH%

pytest . --verbose --maxfail=1000000
if errorlevel 1 exit /b 1
