@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_windows.bat "path\to\song.mp3"
  exit /b 1
)

.\.venv311\Scripts\python.exe main.py "%~1" --output lyric_video.mp4 --device cpu --compute-type int8
pause
