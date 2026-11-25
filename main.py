app = FastAPI(title="NoobNavi AI")
def find_highlights(video_path, total_duration, clip_duration):
    """
    Combines audio peaks and scene changes.
    Finds stronger, separated candidate moments.
    """

    audio_scores = analyze_audio(video_path)
    scene_times = analyze_scenes(video_path)

    candidates = []

    # Calculate audio baseline
    values = [item["score"] for item in audio_scores]

    if values:
        sorted_values = sorted(values)
        baseline = sorted_values[len(sorted_values) // 2]
        high_level = sorted_values[
            min(len(sorted_values) - 1, int(len(sorted_values) * 0.90))
        ]

        audio_range = max(high_level - baseline, 1)

        # Find local audio peaks
        for i, item in enumerate(audio_scores):
            current = item["score"]

            previous = (
                audio_scores[i - 1]["score"] if i > 0 else 0
            )

            following = (
                audio_scores[i + 1]["score"]
                if i < len(audio_scores) - 1 else 0
            )

            is_peak = current >= previous and current >= following

            if is_peak:
                strength = (current - baseline) / audio_range

                if strength > 0.15:
                    candidates.append({
                        "time": item["time"],
                        "score": min(strength, 1.5) * 1.2,
                        "type": "audio"
                    })

    # Add scene changes
    for scene_time in scene_times:
        if 0 <= scene_time <= total_duration:
            candidates.append({
                "time": scene_time,
                "score": 0.8,
                "type": "scene"
            })

    # Combine nearby audio and scene events
    for candidate in candidates:
        for scene_time in scene_times:
            if abs(candidate["time"] - scene_time) <= 4:
                candidate["score"] += 0.35
                break

    # Sort strongest candidates first
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    clip_count = min(
        3,
        max(1, int(total_duration // clip_duration))
    )

    # Keep selected moments separated
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

    # Fallback if the video has very few detectable events
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
