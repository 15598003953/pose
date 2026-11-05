@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
call conda run --no-capture-output -n mmpose python "%~dp0start_integration.py" %*
if errorlevel 1 pause
