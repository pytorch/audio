@echo on

pip install kaldi-io PySoundFile
if errorlevel 1 exit /b 1

REM pytest . --verbose --maxfail=1000000
python test\test_batch_consistency.py -v
python test\test_compliance_kaldi.py -v
python test\test_dataloader.py -v
REM python test\test_datasets.py -v
python test\test_functional.py -v
python test\test_io.py -v
python test\test_kaldi_compatibility.py -v
python test\test_kaldi_io.py -v
python test\test_librosa_compatibility.py -v
python test\test_models.py -v
python test\test_sox_compatibility.py -v
python test\test_sox_effects.py -v
python test\test_torchscript_consistency.py -v
python test\test_transforms.py -v

if errorlevel 1 exit /b 1
