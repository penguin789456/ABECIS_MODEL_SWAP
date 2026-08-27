@echo off
call conda activate CrackPre
pip install -e detectron2 --no-build-isolation
pause