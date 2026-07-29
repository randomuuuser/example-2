"""
Block 2a - Dataset preparation.

Turns the long-form call recordings + test-set JSON into one 16 kHz mono wav
per annotated segment, plus a manifest that every ASR system will consume.

Design decision: the HUMAN LABEL segmentation is canonical.
The JSON carries two segmentations - 'prediction' (produced by whatever ASR
made it) and 'label' (human). They do not share boundaries, so aligning them
by time overlap would inject an arbitrary matching step into the pipeline.
Instead we cut the audio on the label boundaries and re-transcribe those exact
spans with every system. Same audio in, comparable text out, and the WER is
defined without any alignment heuristic.

Expected JSON shape:
    {"VERSION": ..., "samples": {
        "<sample_id>": {
            "sample_id": str,
            "path_audio": str,
            "channel_operator": <index or name>,
            "channel_client":   <index or name>,
            "prediction": {"segments": [...]},
            "label":      {"segments": [
                {"start_time": float, "end_time": float, "channel": ...,
                 "annotation": {"speaker_id":..., "speaker_role":...,
                                "text": str, "words": [...]}}
            ]}
        }, ...
    }}
"""

import json
import os
import subprocess
import wave

import numpy as np

from text_norm import normalize_text

TARGET_SR = 16000

# Normalization settings, frozen once per corpus and persisted next to the
# data. Every later step reloads them, which is what guarantees that the
# reference and all three hypotheses went through the exact same function.
DEFAULT_NORM = {"strip_accents": False, "expand_numbers": True}
NORM_CONFIG_NAME = "norm_config.json"


def write_norm_config(out_dir, norm_config):
    """Persist the normalization settings inside the corpus directory."""
    path = os.path.join(out_dir, NORM_CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(norm_config, handle, indent=1)
    return path


def read_norm_config(out_dir):
    """Reload the settings that were used to prepare this corpus."""
    path = os.path.join(out_dir, NORM_CONFIG_NAME)
    if not os.path.exists(path):
        return dict(DEFAULT_NORM)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Audio I/O
# --------------------------------------------------------------------------

def probe_channels(path):
    """Number of audio channels, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def load_call(path, sr=TARGET_SR):
    """
    Decode a full call to float32, shape [n_samples, n_channels].

    Decoding the whole mp3 once and slicing in memory is far cheaper than one
    ffmpeg seek per segment, and mp3 seeking is imprecise anyway.
    """
    n_channels = probe_channels(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.reshape(-1, n_channels)


def write_wav_mono(path, samples, sr=TARGET_SR):
    """Write a mono 16-bit PCM wav (what NeMo and Whisper both expect)."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------
# Channel resolution
# --------------------------------------------------------------------------

def resolve_channel(segment, sample, n_channels):
    """
    Map a segment's 'channel' field to a column index in the decoded array.

    Accepts either an integer index or a name matching the sample-level
    'channel_operator' / 'channel_client' values. Returns None when the call
    is mono or the channel cannot be resolved, in which case the caller should
    mix down instead of guessing.
    """
    if n_channels == 1:
        return None

    raw = segment.get("channel")
    if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
        return int(raw) if 0 <= int(raw) < n_channels else None

    lookup = {
        str(sample.get("channel_operator")): 0,
        str(sample.get("channel_client")): 1,
    }
    return lookup.get(str(raw))


def extract_segment(call_audio, sr, start_time, end_time, channel_index,
                    pad=0.0):
    """
    Slice one segment, keeping a single channel.

    Isolating the speaker's own channel removes the other party's crosstalk,
    which is the main reason stereo call recordings transcribe better than
    their mixdown.
    """
    start = max(0, int((start_time - pad) * sr))
    end = min(call_audio.shape[0], int((end_time + pad) * sr))
    if end <= start:
        return np.zeros(0, dtype=np.float32)

    chunk = call_audio[start:end]
    if channel_index is None:
        return chunk.mean(axis=1)
    return chunk[:, channel_index]


# --------------------------------------------------------------------------
# JSON parsing
# --------------------------------------------------------------------------

def read_segments(json_path, lang, min_ref_words=10, roles=None,
                  source="label", norm_config=None):
    """
    Flatten the test-set JSON into one record per annotated segment.

    Args:
        json_path: path to the test-set JSON.
        lang: corpus language code ('es', 'de'), applied to every record.
        min_ref_words: drop shorter references. On a 3-word segment the WER
            only takes a handful of discrete values, which is metric noise
            rather than signal. Counted on the NORMALIZED reference, because
            that is what will sit in the WER denominator: in Spanish
            'son 1234 soles' is 3 raw words but 7 normalized ones.
        norm_config: kwargs forwarded to normalize_text. Frozen here and
            persisted, so every later step reuses the identical settings.
        roles: keep only these speaker_role values (e.g. {'Client'}).
        source: 'label' for the human segmentation, 'prediction' to inspect
            the pre-existing ASR one.

    Returns:
        (records, stats) where stats counts what was dropped and why.
    """
    norm_config = dict(DEFAULT_NORM if norm_config is None else norm_config)

    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    records = []
    stats = {"samples": 0, "segments": 0, "empty_text": 0,
             "too_short": 0, "bad_span": 0, "wrong_role": 0, "kept": 0}

    for sample_id, sample in data.get("samples", {}).items():
        stats["samples"] += 1
        segments = sample.get(source, {}).get("segments", []) or []

        for index, segment in enumerate(segments):
            stats["segments"] += 1
            annotation = segment.get("annotation") or {}
            text = (annotation.get("text") or "").strip()
            role = annotation.get("speaker_role")

            if roles is not None and role not in roles:
                stats["wrong_role"] += 1
                continue
            if not text:
                stats["empty_text"] += 1
                continue

            start_time = float(segment.get("start_time", 0.0))
            end_time = float(segment.get("end_time", 0.0))
            if end_time <= start_time:
                stats["bad_span"] += 1
                continue
            reference_norm = normalize_text(text, lang, **norm_config)
            n_ref_words = len(reference_norm.split())
            if n_ref_words < min_ref_words:
                stats["too_short"] += 1
                continue

            records.append({
                "segment_id": f"{sample_id}__{index:04d}",
                "sample_id": sample_id,
                "path_audio": sample.get("path_audio"),
                "channel_raw": segment.get("channel"),
                "channel_operator": sample.get("channel_operator"),
                "channel_client": sample.get("channel_client"),
                "speaker_role": role,
                "speaker_id": annotation.get("speaker_id"),
                "start_time": start_time,
                "end_time": end_time,
                "duration": round(end_time - start_time, 3),
                "reference": text,
                "reference_norm": reference_norm,
                "n_ref_words": n_ref_words,
                "lang": lang,
            })
            stats["kept"] += 1

    return records, stats


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export_segments(records, out_dir, audio_root=""):
    """
    Cut and write one 16 kHz mono wav per record.

    Each call is decoded exactly once. Records already holding a readable
    'wav_path' are skipped, so the step is resumable.
    """
    os.makedirs(out_dir, exist_ok=True)
    by_call = {}
    for record in records:
        by_call.setdefault(record["sample_id"], []).append(record)

    failures = []
    for sample_id, call_records in by_call.items():
        source_path = os.path.join(audio_root, call_records[0]["path_audio"])
        try:
            call_audio = load_call(source_path)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as error:
            failures.append((sample_id, repr(error)))
            continue

        n_channels = call_audio.shape[1]
        for record in call_records:
            wav_path = os.path.join(out_dir, record["segment_id"] + ".wav")
            if os.path.exists(wav_path):
                record["wav_path"] = wav_path
                continue

            channel_index = resolve_channel(
                {"channel": record["channel_raw"]},
                {"channel_operator": record["channel_operator"],
                 "channel_client": record["channel_client"]},
                n_channels,
            )
            samples = extract_segment(
                call_audio, TARGET_SR,
                record["start_time"], record["end_time"], channel_index,
            )
            if samples.size == 0:
                failures.append((record["segment_id"], "empty slice"))
                continue

            write_wav_mono(wav_path, samples)
            record["wav_path"] = wav_path
            record["channel_index"] = channel_index

    return [r for r in records if "wav_path" in r], failures


def write_nemo_manifest(records, manifest_path, task="asr", pnc="yes"):
    """
    Write the JSONL manifest Canary consumes.

    source_lang / target_lang must both be set for an ASR (non-translation)
    task; leaving them out is the usual reason Canary silently falls back to
    English.
    """
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({
                "audio_filepath": os.path.abspath(record["wav_path"]),
                "duration": record["duration"],
                "taskname": task,
                "task": task,
                "source_lang": record["lang"],
                "target_lang": record["lang"],
                "pnc": pnc,
                "segment_id": record["segment_id"],
            }, ensure_ascii=False) + "\n")
    return manifest_path


def write_records(records, path):
    """Persist the flattened records as JSONL for the next blocks."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_records(path):
    """Load records written by write_records."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_corpus(json_path, lang, out_dir, audio_root="",
                   min_ref_words=10, roles=None, norm_config=None):
    """Run the whole block: parse, normalize, cut, export, write manifest."""
    os.makedirs(out_dir, exist_ok=True)
    norm_config = dict(DEFAULT_NORM if norm_config is None else norm_config)
    write_norm_config(out_dir, norm_config)

    records, stats = read_segments(
        json_path, lang, min_ref_words=min_ref_words, roles=roles,
        norm_config=norm_config,
    )
    records, failures = export_segments(records, os.path.join(out_dir, "wav"),
                                        audio_root)
    write_records(records, os.path.join(out_dir, "segments.jsonl"))
    write_nemo_manifest(records, os.path.join(out_dir, "manifest.json"))

    stats["exported"] = len(records)
    stats["failures"] = len(failures)
    total_hours = sum(r["duration"] for r in records) / 3600.0
    stats["hours"] = round(total_hours, 3)
    return records, stats, failures
