
import os
import uuid
import shutil
import threading
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="NoobNavi AI")

BASE_DIR = Path("storage")
UPLOAD_DIR = BASE_DIR / "uploads"
CLIP_DIR = BASE_DIR / "clips"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/clips", StaticFiles(directory=str(CLIP_DIR)), name="clips")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}
jobs_lock = threading.Lock()


@app.get("/")
def home():
    return {
        "message": "Welcome to NoobNavi AI Backend!",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "app": "NoobNavi AI"}


def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def process_video(job_id, input_path, duration):
    try:
        ffmpeg = get_ffmpeg()

        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        probe = subprocess.run(
            [
                ffmpeg,
                "-i", str(input_path),
                "-hide_banner"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        import re

        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            probe.stderr
        )

        if not match:
            raise ValueError("Could not read video duration.")

        hours, minutes, seconds = match.groups()

        total_seconds = (
            int(hours) * 3600 +
            int(minutes) * 60 +
            float(seconds)
        )

        if total_seconds < 1:
            raise ValueError("Video is too short.")

        clip_duration = min(int(duration), int(total_seconds))
        clip_count = min(3, max(1, int(total_seconds // clip_duration)))

        clips = []

        for index in range(clip_count):
            start = index * clip_duration

            if start >= total_seconds:
                break

            output_name = f"{job_id}_highlight_{index + 1}.mp4"
            output_path = CLIP_DIR / output_name

            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-ss", str(start),
                    "-i", str(input_path),
                    "-t", str(clip_duration),
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(output_path)
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=True
            )

            clips.append({
                "name": f"Highlight {index + 1}",
                "url": f"/clips/{output_name}"
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


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    duration: int = Form(30),
    language: str = Form("auto")
):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a valid video.")

    if duration not in [15, 30, 45, 60]:
        raise HTTPException(400, "Invalid clip duration.")

    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}.mp4"

    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(video.file, destination)

        if input_path.stat().st_size == 0:
            raise HTTPException(400, "The uploaded video is empty.")

    except Exception:
        input_path.unlink(missing_ok=True)
        raise

    finally:
        await video.close()

    with jobs_lock:
        jobs[job_id] = {
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
            raise HTTPException(404, "Job not found.")

        return {
            "jobId": job_id,
            "status": job["status"],
            "clips": job["clips"],
            "error": job["error"]
        }


@app.get("/api/clips")
def get_clips():
    with jobs_lock:
        clips = [
            clip
            for job in jobs.values()
            if job["status"] == "completed"
            for clip in job["clips"]
        ]

    return {"clips": clips}
