@echo on

pip install kaldi-io PySoundFile
if errorlevel 1 exit /b 1

REM pytest . --verbose --maxfail=1000000
python test\test_batch_consistency.py
python test\test_compliance_kaldi.py
python test\test_dataloader.py
python test\test_datasets.py
python test\test_functional.py
python test\test_io.py
python test\test_kaldi_compatibility.py
python test\test_kaldi_io.py
python test\test_librosa_compatibility.py
python test\test_models.py
python test\test_sox_compatibility.py
python test\test_sox_effects.py
python test\test_torchscript_consistency.py
python test\test_transforms.py

if errorlevel 1 exit /b 1
