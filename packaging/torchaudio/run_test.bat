@echo on

pip install kaldi-io PySoundFile
if errorlevel 1 exit /b 1

pytest . --verbose --maxfail=1000000
REM python test\test_batch_consistency.py -v
REM python test\test_compliance_kaldi.py -v
REM python test\test_dataloader.py -v
REM python test\test_datasets.py -v
REM python test\test_functional.py -v
REM python test\test_io.py -v
REM python test\test_kaldi_compatibility.py -v
REM python test\test_kaldi_io.py -v
REM python test\test_librosa_compatibility.py -v
REM python test\test_models.py -v
REM python test\test_sox_compatibility.py -v
REM python test\test_sox_effects.py -v
REM python test\test_torchscript_consistency.py -v
REM python test\test_transforms.py -v

if errorlevel 1 exit /b 1
