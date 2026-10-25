@echo off
rem Double-click to run: venv\Scripts\python.exe 2_*.py  (batch replace sign)
cd /d "%~dp0"
for %%F in ("scripts\2_*.py") do "venv\Scripts\python.exe" "%%F"
echo.
pause
