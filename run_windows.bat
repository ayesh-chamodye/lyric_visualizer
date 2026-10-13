@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_windows.bat "path\to\song.mp3"
  exit /b 1
)
python main.py "%~1" --output lyric_video.mp4
pause
