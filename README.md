# Neon Lyric Visualizer (Python)

Create a 1080p MP4 lyric video from a song with automatically estimated word timings, an animated frequency-bar visualizer, pulsing neon background, and original audio.

## Requirements

- Python 3.10–3.12 recommended
- FFmpeg installed and available as `ffmpeg` in your terminal
- Internet connection for the first run, because Faster-Whisper downloads its speech-recognition model
- For CUDA acceleration, a compatible NVIDIA driver and compatible CUDA libraries. CPU mode works too but can be much slower.

## Windows setup (PowerShell)

```powershell
cd path\to\lyric_visualizer
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg with winget:

```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell after installing FFmpeg, then verify:

```powershell
ffmpeg -version
```

## Linux (Arch Linux)

```bash
sudo pacman -S ffmpeg python python-pip python-virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you have NVIDIA CUDA set up, try `--device cuda --compute-type float16`; otherwise use the default CPU mode.

## Run

```bash
python main.py "song.mp3" --output "my_lyric_video.mp4"
```

Optional arguments:

```bash
python main.py "song.mp3" --output "my_lyric_video.mp4" --model small --language en
python main.py "song.mp3" --output "my_lyric_video.mp4" --model medium --device cuda --compute-type float16
python main.py "song.mp3" --output "preview.mp4" --width 1280 --height 720 --fps 24
```

Model choices: `tiny`, `base`, `small`, `medium`, `large-v3`. Larger models usually improve transcription, but require more memory and time. Leave `--language` out for automatic language detection; use a language code such as `en`, `si`, or `ta` when you know the song language.

## Output

The final file is H.264 video + AAC audio in an MP4 container, with `yuv420p` pixel format for broad player compatibility. The output file is created at the path supplied to `--output`.

## Accuracy notes

- Word-level timestamps are estimated by the speech-recognition model; they are not guaranteed to match every sung syllable exactly.
- Instrumental music, backing vocals, heavy effects, fast rap, and non-English songs may produce incorrect words or timings. Try a larger model and specify `--language`.
- This program automatically estimates timings; it does not promise perfect karaoke-grade forced alignment. For a public release, review the lyrics and timing before publishing.
- Only process audio you have permission to use. YouTube may block or monetize videos containing copyrighted songs according to the rights holder's policies.

## Troubleshooting

- **`ffmpeg not found`:** install FFmpeg, reopen the terminal, and run `ffmpeg -version`.
- **CUDA error:** omit CUDA options and use the default CPU mode, or install a compatible CUDA/cuDNN setup for CTranslate2.
- **Not enough RAM/VRAM:** use `--model base` or `--model tiny`.
- **Poor lyrics:** use `--model medium` or `--model large-v3`, specify `--language`, or try a clearer vocal mix.
