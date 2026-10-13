#!/usr/bin/env python3
"""Neon lyric-video generator with automatic transcription and audio-reactive visuals.

Requirements: FFmpeg installed and on PATH; pip install -r requirements.txt
Usage: python main.py path/to/song.mp3 --output lyric_video.mp4
"""
from __future__ import annotations
import argparse, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1920, 1080, 30
BG = (7, 10, 24)
CYAN = (70, 220, 255)
WHITE = (245, 249, 255)
MUTED = (135, 157, 190)


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("FFmpeg was not found. Install it and make sure 'ffmpeg' works in your terminal.")
    return exe


def extract_audio(input_path: Path, wav_path: Path, ffmpeg: str) -> None:
    cmd = [ffmpeg, "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav_path)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("Could not decode audio/video input:\n" + result.stderr[-2500:])


def transcribe_words(wav_path: Path, model_size: str, device: str, compute_type: str, language: str | None):
    print(f"Loading Whisper model '{model_size}' ({device}/{compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(wav_path), language=language, word_timestamps=True,
        vad_filter=False, condition_on_previous_text=True, beam_size=5
    )
    words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                token = (word.word or "").strip()
                if token and word.start is not None and word.end is not None:
                    words.append({"start": max(0.0, float(word.start)), "end": max(float(word.start) + .04, float(word.end)), "text": token})
        elif segment.text.strip():
            words.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()})
    if not words:
        raise RuntimeError("No lyrics were detected. Try another model or check that the audio contains vocals.")
    print(f"Detected language: {info.language} ({info.language_probability:.0%} confidence); timed tokens: {len(words)}")
    return words


def get_font(size: int, bold: bool = False):
    candidates = []
    if sys.platform == "win32":
        candidates += [r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
                       r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"]
    elif sys.platform == "darwin":
        candidates += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"]
    candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def audio_features(wav_path: Path, duration: float, fps: int):
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    samples_per_frame = max(1, int(sr / fps))
    frame_count = int(math.ceil(duration * fps))
    rms = np.zeros(frame_count, dtype=np.float32)
    spectrum = np.zeros((frame_count, 48), dtype=np.float32)
    for i in range(frame_count):
        chunk = audio[i * samples_per_frame:(i + 1) * samples_per_frame]
        if len(chunk):
            rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-9)
            # Slightly longer FFT window gives a more stable visualizer.
            a = max(0, i * samples_per_frame - samples_per_frame)
            b = min(len(audio), i * samples_per_frame + samples_per_frame * 2)
            fft = np.abs(np.fft.rfft(audio[a:b] * np.hanning(max(1, b-a)))) if b > a else np.zeros(1)
            if len(fft) > 1:
                bins = np.array_split(fft[1:], 48)
                spectrum[i] = [float(np.mean(x)) if len(x) else 0 for x in bins]
    if rms.max() > 0:
        rms = np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1.5)
    for i in range(frame_count):
        m = spectrum[i].max()
        if m > 0:
            spectrum[i] /= m
    return rms, spectrum, frame_count


def draw_text_with_glow(frame: np.ndarray, text: str, center: tuple[int, int], font, color=(245,249,255), glow=(30,180,255), max_width=1600):
    # Render text into a transparent layer, glow it, then composite onto the frame.
    base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    if bbox[2] - bbox[0] > max_width:
        # Shrink long lyrics so they remain inside the safe area.
        size = getattr(font, "size", 56)
        while size > 28 and bbox[2] - bbox[0] > max_width:
            size -= 2
            font = get_font(size, True)
            bbox = draw.textbbox((0, 0), text, font=font)
    x = int(center[0] - (bbox[2] - bbox[0]) / 2 - bbox[0])
    y = int(center[1] - (bbox[3] - bbox[1]) / 2 - bbox[1])
    draw.text((x, y), text, font=font, fill=(*color, 255))
    glow_layer = layer.filter(ImageFilter.GaussianBlur(20))
    # Tint the blurred alpha to cyan.
    alpha = glow_layer.getchannel("A")
    tint = Image.new("RGBA", base.size, (*glow, 0))
    tint.putalpha(alpha.point(lambda p: int(p * .75)))
    base = Image.alpha_composite(base, tint)
    base = Image.alpha_composite(base, layer)
    return cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def lyric_lines(words, t: float, max_words=7):
    if not words:
        return "", ""
    idx = -1
    for i, word in enumerate(words):
        if word["start"] <= t < word["end"] + 0.08:
            idx = i
            break
        if word["start"] <= t:
            idx = i
        if word["start"] > t:
            break
    if idx < 0:
        return "", ""
    # Prefer a short line around the active word, stopping at punctuation or a long pause.
    start = idx
    while start > 0 and idx - start < max_words // 2:
        prev, cur = words[start-1], words[start]
        if cur["start"] - prev["end"] > 0.75 or any(p in prev["text"] for p in ".!?;:"):
            break
        start -= 1
    end = idx
    while end + 1 < len(words) and end - start + 1 < max_words:
        cur, nxt = words[end], words[end+1]
        if nxt["start"] - cur["end"] > 0.75 or any(p in cur["text"] for p in ".!?;:"):
            break
        end += 1
    current = " ".join(w["text"] for w in words[start:end+1]).strip()
    # A subtle upcoming line adds karaoke-like context.
    next_line = ""
    if end + 1 < len(words) and words[end+1]["start"] - words[end]["end"] < 1.2:
        nxt_end = min(len(words)-1, end + max_words)
        next_line = " ".join(w["text"] for w in words[end+1:nxt_end+1]).strip()
        for punct in ".!?;:":
            if punct in next_line:
                next_line = next_line.split(punct)[0] + punct
                break
    return current, next_line


def render_video(input_path: Path, output_path: Path, words, width: int, height: int, fps: int, wav_path: Path, ffmpeg: str):
    global W, H
    W, H = width, height
    audio, sr = sf.read(str(wav_path), dtype="float32")
    duration = len(audio) / sr
    rms, spectrum, frames = audio_features(wav_path, duration, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + "_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the video writer.")
    title_font = get_font(max(24, int(H * .025)), True)
    lyric_font = get_font(max(38, int(H * .064)), True)
    next_font = get_font(max(22, int(H * .027)), False)
    small_font = get_font(max(18, int(H * .019)), False)
    print(f"Rendering {frames} frames at {fps} FPS ({W}x{H})...")
    # Precompute the background radial field once; recomputing a full HD grid per frame is wasteful.
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * .5, H * .48
    radial = np.exp(-(((xx-cx)/(W*.48))**2 + ((yy-cy)/(H*.58))**2) * 3.0)
    del xx, yy
    try:
        for fi in range(frames):
            t = fi / fps
            energy = float(rms[min(fi, len(rms)-1)])
            spec = spectrum[min(fi, len(spectrum)-1)]
            # Dark blue/purple gradient with a gently pulsing center glow.
            pulse = .45 + .55 * energy
            frame = np.zeros((H, W, 3), dtype=np.uint8)
            frame[:,:,0] = np.clip(BG[0] + radial * 23 * pulse, 0, 255).astype(np.uint8)
            frame[:,:,1] = np.clip(BG[1] + radial * 18 * pulse, 0, 255).astype(np.uint8)
            frame[:,:,2] = np.clip(BG[2] + radial * 52 * pulse, 0, 255).astype(np.uint8)
            # Ambient orbit rings.
            center = (W//2, int(H*.48))
            for j, radius in enumerate((int(H*.21), int(H*.29), int(H*.37))):
                cv2.ellipse(frame, center, (int(radius*1.65), radius), -8, 0, 360,
                            (25 + int(energy*25), 45 + int(energy*60), 80 + int(energy*100)), 1, cv2.LINE_AA)
            # 48 frequency bars along the lower region.
            bar_area_w = W * .82
            bar_gap = max(3, int(W*.004))
            bar_w = max(3, int((bar_area_w - 47*bar_gap)/48))
            x0 = int((W - (48*bar_w + 47*bar_gap))/2)
            baseline = int(H*.82)
            max_h = int(H*.13)
            for j, val in enumerate(spec):
                # Mirror a little movement and lift quiet bins so the spectrum stays visible.
                height_bar = int((.10 + .90*float(val)) * max_h * (.35 + .65*energy))
                x = x0 + j*(bar_w+bar_gap)
                y = baseline-height_bar
                color = (min(255, 110+int(val*100)), min(255, 100+int(val*120)), 255)
                cv2.rectangle(frame, (x, y), (x+bar_w, baseline), color, -1, cv2.LINE_AA)
                # Small reflected bars for a polished visualizer look.
                cv2.rectangle(frame, (x, baseline+5), (x+bar_w, baseline+5+int(height_bar*.16)), (35, 65, 100), -1)
            # Header and progress indicator.
            cv2.putText(frame, "LYRIC / VISUALIZER", (int(W*.055), int(H*.085)), cv2.FONT_HERSHEY_SIMPLEX, H/1080*.8, (170,205,230), 1, cv2.LINE_AA)
            cv2.putText(frame, "AUTO-SYNC  •  AUDIO REACTIVE", (int(W*.055), int(H*.115)), cv2.FONT_HERSHEY_SIMPLEX, H/1080*.42, (100,145,180), 1, cv2.LINE_AA)
            progress = min(1.0, t / max(duration, .01))
            cv2.rectangle(frame, (int(W*.055), int(H*.93)), (int(W*.945), int(H*.934)), (35, 50, 75), -1)
            cv2.rectangle(frame, (int(W*.055), int(H*.93)), (int(W*(.055+.89*progress)), int(H*.934)), (255, 190, 80), -1)
            # Lyric line with soft glow and upcoming line.
            current, upcoming = lyric_lines(words, t)
            if current:
                frame = draw_text_with_glow(frame, current, (W//2, int(H*.49)), lyric_font, WHITE, (25, 150, 255), int(W*.86))
            if upcoming:
                # Draw upcoming lyric subdued; smaller font.
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
                d = ImageDraw.Draw(pil)
                box = d.textbbox((0,0), upcoming, font=next_font)
                if box[2]-box[0] > W*.84:
                    upcoming = upcoming[:max(10, int(len(upcoming)*.84*W/(box[2]-box[0]))-3)] + "…"
                    box = d.textbbox((0,0), upcoming, font=next_font)
                d.text(((W-(box[2]-box[0]))/2, H*.58), upcoming, font=next_font, fill=(*MUTED, 230))
                frame = cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            # Small live energy meter.
            cv2.putText(frame, f"ENERGY {int(min(100, energy*70)):02d}%", (int(W*.80), int(H*.085)), cv2.FONT_HERSHEY_SIMPLEX, H/1080*.48, (100,190,225), 1, cv2.LINE_AA)
            writer.write(frame)
            if fi % (fps*10) == 0:
                print(f"  {t:7.1f}s / {duration:.1f}s ({progress:.0%})")
    finally:
        writer.release()
    print("Muxing original audio and encoding final MP4...")
    cmd = [ffmpeg, "-y", "-i", str(silent_path), "-i", str(input_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output_path)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("FFmpeg export failed:\n" + result.stderr[-3000:])
    try:
        silent_path.unlink()
    except OSError:
        pass
    print(f"\nDone! MP4 saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Create an audio-reactive lyric video with automatic lyric timing.")
    parser.add_argument("audio", help="Input audio/video file (MP3, WAV, M4A, MP4, etc.)")
    parser.add_argument("--output", default="lyric_video.mp4", help="Output MP4 path (default: lyric_video.mp4)")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size; larger is more accurate but slower")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--compute-type", default="int8", help="Examples: int8 (CPU), float16 (CUDA), int8_float16 (CUDA)")
    parser.add_argument("--language", default=None, help="Optional language code, e.g. en, si, ta; auto-detect if omitted")
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()
    source = Path(args.audio).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Input file does not exist: {source}")
    if source == output:
        parser.error("Output path must be different from input path.")
    ffmpeg = require_ffmpeg()
    device = args.device
    compute_type = args.compute_type
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            compute_type = "float16" if device == "cuda" and args.compute_type == "int8" else args.compute_type
        except Exception:
            device, compute_type = "cpu", args.compute_type
    with tempfile.TemporaryDirectory(prefix="lyric_video_") as temp_dir:
        wav_path = Path(temp_dir) / "analysis.wav"
        print("Extracting audio for transcription and visual analysis...")
        extract_audio(source, wav_path, ffmpeg)
        words = transcribe_words(wav_path, args.model, device, compute_type, args.language)
        render_video(source, output, words, args.width, args.height, args.fps, wav_path, ffmpeg)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
