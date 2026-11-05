@echo off
"%~dp0PoseIntegration.exe" --check
if errorlevel 1 goto finished
"%~dp0PoseIntegration.exe" --check-models
:finished
pause
