
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

# =====================================
# NOOBNAVI AI - BACKEND
# =====================================

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

app.mount(
    "/clips",
    StaticFiles(directory=str(CLIP_DIR)),
    name="clips"
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

jobs = {}
jobs_lock = threading.Lock()


# =====================================
# VIDEO DURATION
# =====================================

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


# =====================================
# AUDIO ANALYSIS
# =====================================

def analyze_audio(video_path):

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
        return []

    finally:

        if audio_path.exists():
            audio_path.unlink()


# =====================================
# SCENE CHANGE ANALYSIS
# =====================================

def analyze_scenes(video_path):

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


# =====================================
# SMART HIGHLIGHT DETECTION
# =====================================

def find_highlights(video_path, total_duration, clip_duration):

    audio_scores = analyze_audio(video_path)
    scene_times = analyze_scenes(video_path)

    candidates = []

    values = [
        item["score"]
        for item in audio_scores
    ]

    if values:

        sorted_values = sorted(values)
        baseline = sorted_values[len(sorted_values) // 2]

        high_level = sorted_values[
            min(
                len(sorted_values) - 1,
                int(len(sorted_values) * 0.90)
            )
        ]

        audio_range = max(high_level - baseline, 1)

        for i, item in enumerate(audio_scores):

            current = item["score"]

            previous = (
                audio_scores[i - 1]["score"]
                if i > 0 else 0
            )

            following = (
                audio_scores[i + 1]["score"]
                if i < len(audio_scores) - 1 else 0
            )

            if current >= previous and current >= following:

                strength = (current - baseline) / audio_range

                if strength > 0.15:

                    candidates.append({
                        "time": item["time"],
                        "score": min(strength, 1.5) * 1.2,
                        "type": "audio"
                    })

    for scene_time in scene_times:

        if 0 <= scene_time <= total_duration:

            candidates.append({
                "time": scene_time,
                "score": 0.8,
                "type": "scene"
            })

    for candidate in candidates:

        for scene_time in scene_times:

            if abs(candidate["time"] - scene_time) <= 4:

                candidate["score"] += 0.35
                break

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    clip_count = min(
        3,
        max(1, int(total_duration // clip_duration))
    )

    minimum_gap = max(clip_duration * 0.75, 10)

    selected = []

    for candidate in candidates:

        moment = candidate["time"]

        if any(
            abs(moment - existing) < minimum_gap
            for existing in selected
        ):
            continue

        selected.append(moment)

        if len(selected) >= clip_count:
            break

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


# =====================================
# CREATE NORMAL VIDEO CLIP
# =====================================

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


# =====================================
# CREATE VERTICAL SHORTS / REELS
# =====================================

def create_vertical_clip(input_path, output_path, start_time, duration):

    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920",
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


# =====================================
# PROCESS NORMAL VIDEO
# =====================================

def process_video(job_id, input_path, duration):

    try:

        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        total_duration = get_video_duration(input_path)

        if total_duration <= 0:
            raise ValueError("Video is empty.")

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
                "url": f"/clips/{filename}",
                "format": "original"
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


# =====================================
# PROCESS VERTICAL VIDEO
# =====================================

def process_vertical_video(job_id, input_path, duration):

    try:

        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        total_duration = get_video_duration(input_path)

        if total_duration <= 0:
            raise ValueError("Video is empty.")

        actual_duration = min(
            float(duration),
            total_duration
        )

        highlights = find_highlights(
            input_path,
            total_duration,
            actual_duration
        )

        clips = []

        for index, center in enumerate(highlights, start=1):

            start_time = max(
                0,
                min(
                    center - actual_duration / 2,
                    total_duration - actual_duration
                )
            )

            filename = f"{job_id}_vertical_{index}.mp4"
            output_path = CLIP_DIR / filename

            create_vertical_clip(
                input_path,
                output_path,
                start_time,
                actual_duration
            )

            clips.append({
                "name": f"Vertical Short {index}",
                "url": f"/clips/{filename}",
                "format": "vertical_9_16",
                "aspect_ratio": "9:16"
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


# =====================================
# HOME AND HEALTH
# =====================================

@app.get("/")
def home():

    return {
        "message": "Welcome to NoobNavi AI Backend!",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =====================================
# UPLOAD NORMAL VIDEO
# =====================================

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
            "language": language,
            "format": "original"
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


# =====================================
# UPLOAD VERTICAL VIDEO
# =====================================

@app.post("/api/vertical-jobs")
async def create_vertical_job(
    video: UploadFile = File(...),
    duration: int = Form(30)
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
    input_path = UPLOAD_DIR / f"{job_id}_vertical_input.mp4"

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
            "format": "vertical_9_16"
        }

    thread = threading.Thread(
        target=process_vertical_video,
        args=(job_id, input_path, duration),
        daemon=True
    )

    thread.start()

    return {
        "jobId": job_id,
        "status": "queued",
        "format": "vertical_9_16"
    }


# =====================================
# CHECK JOB STATUS
# =====================================

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


# =====================================
# GET ALL CLIPS
# =====================================

@app.get("/api/clips")
def get_clips():

    all_clips = []

    with jobs_lock:

        for job in jobs.values():

            if job["status"] == "completed":
                all_clips.extend(job["clips"])

    return {
        "clips": all_clips
    }
