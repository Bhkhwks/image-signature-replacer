@echo off
rem Double-click to run: venv\Scripts\python.exe 0_*.py  (auto upright images)
cd /d "%~dp0"
for %%F in ("scripts\0_*.py") do "venv\Scripts\python.exe" "%%F"
echo.
pause
