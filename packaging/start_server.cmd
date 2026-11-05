@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
"%~dp0PoseIntegration.exe" --headless --config "%~dp0configs\integration_config.json" %*
if errorlevel 1 pause
