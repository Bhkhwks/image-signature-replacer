@echo off
rem Double-click to run: venv\Scripts\python.exe 1_*.py  (draw mask boxes)
cd /d "%~dp0"
for %%F in ("scripts\1_*.py") do "venv\Scripts\python.exe" "%%F"
echo.
pause
