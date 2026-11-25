
import os
import re
import uuid
import threading
import subprocess
import array
from pathlib import Path

import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# -----------------------------
# NoobNavi AI Backend
# -----------------------------

app = FastAPI(title="NoobNavi AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
CLIP_DIR = BASE_DIR / "storage" / "clips"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/clips", StaticFiles(directory=str(CLIP_DIR)), name="clips")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

jobs = {}
jobs_lock = threading.Lock()


# -----------------------------
# Video information
# -----------------------------

def get_video_duration(video_path):
    result = subprocess.run(
        [FFMPEG, "-i", str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr
    )

    if not match:
        raise ValueError("Could not read video duration.")

    hours, minutes, seconds = match.groups()

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


# -----------------------------
# Audio analysis
# -----------------------------

def analyze_audio(video_path):
    """
    Measures audio energy in two-second windows.
    Louder sections become possible highlight candidates.
    """

    audio_path = UPLOAD_DIR / f"{uuid.uuid4()}_audio.raw"

    try:
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "8000",
                "-f", "s16le",
                str(audio_path)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )

        raw_data = audio_path.read_bytes()

        samples = array.array("h")
        samples.frombytes(raw_data)

        # Two seconds at 8000 samples per second
        window_size = 16000

        scores = []

        for start in range(0, len(samples), window_size):
            window = samples[start:start + window_size]

            if len(window) < 4000:
                continue

            energy = sum(sample * sample for sample in window)
            rms = (energy / len(window)) ** 0.5

            scores.append({
                "time": (start + len(window) / 2) / 8000,
                "score": rms
            })

        return scores

    except Exception:
        # Some videos may not contain audio.
        return []

    finally:
        if audio_path.exists():
            audio_path.unlink()


# -----------------------------
# Scene change detection
# -----------------------------

def analyze_scenes(video_path):
    """
    Detects noticeable visual scene changes using FFmpeg.
    """

    try:
        result = subprocess.run(
            [
                FFMPEG,
                "-i", str(video_path),
                "-vf", "select='gt(scene,0.30)',showinfo",
                "-an",
                "-f", "null",
                "-"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        times = re.findall(
            r"pts_time:([0-9]+(?:\.[0-9]+)?)",
            result.stderr
        )

        return [float(value) for value in times]

    except Exception:
        return []


# -----------------------------
# Smart highlight selection
# -----------------------------

def find_highlights(video_path, total_duration, clip_duration):

    audio_scores = analyze_audio(video_path)
    scene_times = analyze_scenes(video_path)

    candidates = []

    max_audio = max(
        [item["score"] for item in audio_scores],
        default=0
    )

    # Audio-based candidates
    for item in audio_scores:
        normalized_audio = (
            item["score"] / max_audio if max_audio > 0 else 0
        )

        candidates.append({
            "time": item["time"],
            "score": normalized_audio
        })

    # Add extra importance around scene changes
    for scene_time in scene_times:
        if scene_time < total_duration:
            candidates.append({
                "time": scene_time,
                "score": 1.2
            })

    # If audio and scene analysis find nothing,
    # use evenly spaced fallback positions.
    if not candidates:
        candidates = [
            {
                "time": total_duration * position / 4,
                "score": 0.5
            }
            for position in [1, 2, 3]
        ]

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    clip_count = min(
        3,
        max(1, int(total_duration // clip_duration))
    )

    minimum_gap = max(clip_duration * 0.6, 8)

    selected = []

    for candidate in candidates:
        center = candidate["time"]

        if center < 0 or center > total_duration:
            continue

        # Avoid selecting overlapping highlight moments.
        if any(
            abs(center - existing) < minimum_gap
            for existing in selected
        ):
            continue

        selected.append(center)

        if len(selected) >= clip_count:
            break

    # Fill missing positions if necessary.
    fallback_positions = [
        total_duration * 0.25,
        total_duration * 0.50,
        total_duration * 0.75
    ]

    for position in fallback_positions:
        if len(selected) >= clip_count:
            break

        if not any(
            abs(position - existing) < minimum_gap
            for existing in selected
        ):
            selected.append(position)

    selected.sort()

    return selected[:clip_count]


# -----------------------------
# Create video clips
# -----------------------------

def create_clip(input_path, output_path, start_time, duration):

    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "25",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True
    )


# -----------------------------
# Background processing
# -----------------------------

def process_video(job_id, input_path, duration):

    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        total_duration = get_video_duration(input_path)

        if total_duration <= 0:
            raise ValueError("The uploaded video is empty.")

        actual_clip_duration = min(
            float(duration),
            total_duration
        )

        highlights = find_highlights(
            input_path,
            total_duration,
            actual_clip_duration
        )

        clips = []

        for index, center in enumerate(highlights, start=1):

            start_time = max(
                0,
                min(
                    center - actual_clip_duration / 2,
                    total_duration - actual_clip_duration
                )
            )

            filename = f"{job_id}_highlight_{index}.mp4"
            output_path = CLIP_DIR / filename

            create_clip(
                input_path,
                output_path,
                start_time,
                actual_clip_duration
            )

            clips.append({
                "name": f"Highlight {index}",
                "url": f"/clips/{filename}"
            })

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["clips"] = clips

    except Exception as error:

        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(error)

    finally:
        if input_path.exists():
            input_path.unlink()


# -----------------------------
# API endpoints
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to NoobNavi AI Backend!",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    duration: int = Form(30),
    language: str = Form("auto")
):

    if duration not in [15, 30, 45, 60]:
        raise HTTPException(
            status_code=400,
            detail="Invalid clip duration."
        )

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a video."
        )

    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}.mp4"

    try:
        with input_path.open("wb") as file:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                file.write(chunk)

    finally:
        await video.close()

    if input_path.stat().st_size == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty."
        )

    with jobs_lock:
        jobs[job_id] = {
            "jobId": job_id,
            "status": "queued",
            "clips": [],
            "error": None,
            "language": language
        }

    thread = threading.Thread(
        target=process_video,
        args=(job_id, input_path, duration),
        daemon=True
    )

    thread.start()

    return {
        "jobId": job_id,
        "status": "queued"
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):

    with jobs_lock:
        job = jobs.get(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail="Job not found."
            )

        return dict(job)


@app.get("/api/clips")
def get_clips():

    all_clips = []

    with jobs_lock:
        for job in jobs.values():
            if job["status"] == "completed":
                all_clips.extend(job["clips"])

    return {"clips": all_clips}
