@echo off
rem Builds dist\ChestLab.exe (one file). Needs the project's .venv with requirements.txt and pyinstaller installed.
cd /d %~dp0
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller --onefile --noconfirm --clean --name ChestLab --distpath dist --workpath build --specpath build --paths "%~dp0." --add-data "%~dp0assets;assets" --collect-submodules winrt --collect-all winocr --hidden-import keyboard --hidden-import chestlab.procs live_app.py
.venv\Scripts\python.exe -c "print('built dist\ChestLab.exe: run it with --selftest to check it (opens no window)')"
