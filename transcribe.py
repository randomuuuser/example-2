"""
Block 2b - Transcription with the target model and the proxies.

Three systems run over the SAME wav slices produced by block 2a:
    target  : nvidia/canary-180m-flash   (the model whose WER we predict)
    proxy A : openai/whisper-large-v3    (different family -> decorrelated errors)
    proxy B : nvidia/canary-1b-v2        (same family -> partially correlated)

Each system writes its own JSONL cache, so a crashed run resumes and a new
proxy can be added without recomputing the others. Models are loaded and
released one at a time - never hold three ASR systems on the GPU at once.

Run order:
    1. sanity_check_canary(...)   <- do this FIRST, see the note below
    2. run_canary(...) for the target and for proxy B
    3. run_whisper(...) for proxy A

IMPORTANT - verify multilingual behaviour before anything else.
There is at least one community report of canary-180m-flash returning English
for non-English audio while canary-1b-flash worked on the same input. The usual
cause is source_lang / target_lang not reaching the model. Look at the raw
output of ~20 es and ~20 de segments with your own eyes before building
features on top of it.
"""

import gc
import json
import os


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def load_cache(path):
    """Read a transcription cache into {segment_id: record}."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return {r["segment_id"]: r for r in map(json.loads, handle) if r}


def append_cache(path, rows):
    """Append transcription rows to the cache."""
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pending(records, cache):
    """Records not yet transcribed by this system."""
    return [r for r in records if r["segment_id"] not in cache]


def release_model(model):
    """Free GPU memory between systems."""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Canary (NeMo)
# --------------------------------------------------------------------------

def load_canary(model_name="nvidia/canary-180m-flash", beam_size=1):
    """
    Load a Canary checkpoint and force greedy decoding.

    Greedy is what NVIDIA uses for the model-card WER numbers, and beam search
    would change the error profile we are trying to predict.
    """
    from nemo.collections.asr.models import EncDecMultiTaskModel

    model = EncDecMultiTaskModel.from_pretrained(model_name)
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = beam_size
    model.change_decoding_strategy(decode_cfg)
    model.eval()
    return model


def hypothesis_text(item):
    """
    Extract the text from one NeMo transcribe() result.

    NeMo returns plain strings on some versions and Hypothesis objects on
    others, and occasionally a (best, all) tuple. Normalize all three.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and item:
        return hypothesis_text(item[0])
    return getattr(item, "text", "") or ""


def run_canary(records, cache_path, model_name="nvidia/canary-180m-flash",
               batch_size=16, chunk=256):
    """
    Transcribe every pending record with a Canary model.

    Writes a temporary manifest per chunk so progress survives a crash. The
    manifest carries source_lang / target_lang per segment, which is what makes
    the multilingual path work.
    """
    cache = load_cache(cache_path)
    todo = pending(records, cache)
    if not todo:
        return cache

    model = load_canary(model_name)
    manifest_path = cache_path + ".tmp_manifest.json"

    try:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for record in batch:
                    handle.write(json.dumps({
                        "audio_filepath": os.path.abspath(record["wav_path"]),
                        "duration": record["duration"],
                        "taskname": "asr",
                        "task": "asr",
                        "source_lang": record["lang"],
                        "target_lang": record["lang"],
                        "pnc": "yes",
                    }, ensure_ascii=False) + "\n")

            outputs = model.transcribe(manifest_path, batch_size=batch_size)
            rows = [
                {"segment_id": record["segment_id"],
                 "system": model_name,
                 "text": hypothesis_text(item)}
                for record, item in zip(batch, outputs)
            ]
            append_cache(cache_path, rows)
            for row in rows:
                cache[row["segment_id"]] = row
            print(f"  {model_name}: {min(start + chunk, len(todo))}/{len(todo)}")
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        release_model(model)

    return cache


# --------------------------------------------------------------------------
# Whisper (transformers)
# --------------------------------------------------------------------------

def run_whisper(records, cache_path, model_name="openai/whisper-large-v3",
                batch_size=8):
    """
    Transcribe every pending record with a Whisper model, as proxy A.

    Greedy decoding and an explicit language token: letting Whisper detect the
    language itself adds a failure mode that has nothing to do with the WER we
    are trying to predict.
    """
    import torch
    from transformers import pipeline

    cache = load_cache(cache_path)
    todo = pending(records, cache)
    if not todo:
        return cache

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model_name,
                   device=device, torch_dtype=dtype)

    try:
        by_lang = {}
        for record in todo:
            by_lang.setdefault(record["lang"], []).append(record)

        for lang, batch in by_lang.items():
            paths = [r["wav_path"] for r in batch]
            outputs = asr(
                paths,
                batch_size=batch_size,
                generate_kwargs={"language": lang, "task": "transcribe",
                                 "num_beams": 1},
            )
            rows = [
                {"segment_id": record["segment_id"],
                 "system": model_name,
                 "text": (out.get("text") or "").strip()}
                for record, out in zip(batch, outputs)
            ]
            append_cache(cache_path, rows)
            for row in rows:
                cache[row["segment_id"]] = row
            print(f"  {model_name} [{lang}]: {len(rows)} segments")
    finally:
        release_model(asr)

    return cache


# --------------------------------------------------------------------------
# Sanity check - run this before anything else
# --------------------------------------------------------------------------

def sanity_check_canary(records, model_name="nvidia/canary-180m-flash",
                        n_per_lang=20):
    """
    Transcribe a handful of segments per language and print them next to the
    reference, so the multilingual path can be verified by eye.

    What you are looking for: output in the SAME language as the reference. If
    Spanish audio comes back as English text, stop and fix the manifest before
    spending a GPU-hour on the full corpus.
    """
    from collections import Counter

    sample = []
    per_lang = Counter()
    for record in records:
        if per_lang[record["lang"]] < n_per_lang:
            sample.append(record)
            per_lang[record["lang"]] += 1

    cache_path = "sanity_canary.jsonl"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    cache = run_canary(sample, cache_path, model_name=model_name)

    for record in sample:
        hypothesis = cache.get(record["segment_id"], {}).get("text", "")
        print(f"\n[{record['lang']}] {record['segment_id']}")
        print(f"  REF : {record['reference']}")
        print(f"  HYP : {hypothesis}")

    return cache


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

SYSTEMS = {
    "target": ("nvidia/canary-180m-flash", "canary"),
    "proxy_a": ("openai/whisper-large-v3", "whisper"),
    "proxy_b": ("nvidia/canary-1b-v2", "canary"),
}


def run_all(records, out_dir, systems=None):
    """
    Run every system in turn and return {role: {segment_id: text}}.

    Models are loaded sequentially, so peak GPU memory is that of the largest
    single system rather than their sum.
    """
    os.makedirs(out_dir, exist_ok=True)
    systems = systems or SYSTEMS
    results = {}

    for role, (model_name, kind) in systems.items():
        cache_path = os.path.join(out_dir, f"{role}.jsonl")
        print(f"\n=== {role}: {model_name} ===")
        runner = run_canary if kind == "canary" else run_whisper
        cache = runner(records, cache_path, model_name=model_name)
        results[role] = {sid: row["text"] for sid, row in cache.items()}

    return results
