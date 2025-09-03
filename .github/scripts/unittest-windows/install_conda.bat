Start-Process -FilePath %miniconda_exe% -ArgumentList "/S /InstallationType=AllUsers /AddToPath=0 /D=%tmp_conda%" -NoNewWindow -Wait
