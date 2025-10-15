@echo off
call .venv\Scripts\activate
pyinstaller --onefile --windowed --icon="icon.ico" StableDiffuctionBatchRunner.py
pause
