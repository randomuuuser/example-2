"""
Pipeline steps, meant to be called from a notebook one cell at a time.

Each function does one thing, prints a short report, and RETURNS the object
you will want to inspect next. Replaces the argparse version of main.py.

    import pipeline as pl

    records = pl.prepare("peru/testset.json", lang="es",
                         out_dir="data/peru", audio_root="/data/peru")
    pl.sanity(records)                          # eyeball Canary, 20 seg/lang
    pl.transcribe(records, "data/peru", ["target"])
    pl.accents(records, "data/peru")            # decide strip_accents
    pl.transcribe(records, "data/peru")         # the two proxies
    rows = pl.merge(records, "data/peru")
    df = pl.to_frame(rows)

Order matters. prepare() freezes the normalization settings into
norm_config.json; every later step reloads them, so the reference and all
hypotheses are provably normalized the same way. sanity() runs before the full
transcription because a Canary language misconfiguration costs 30 seconds to
catch on 40 segments and a GPU-hour to discover afterwards. accents() needs
the target transcriptions, hence the two-stage transcribe.
"""

import os

from merge import build_table
from prepare_dataset import prepare_corpus, read_norm_config, read_records
from text_norm import diagnose_accent_impact
from transcribe import SYSTEMS, load_cache, run_all, sanity_check_canary


def report(title, pairs):
    """Print a compact key: value block."""
    print(f"\n{title}")
    width = max((len(str(k)) for k, _ in pairs), default=0)
    for key, value in pairs:
        print(f"  {str(key):<{width}} : {value}")


def load(out_dir):
    """Reload the records written by prepare()."""
    return read_records(os.path.join(out_dir, "segments.jsonl"))


def prepare(json_path, lang, out_dir, audio_root="", min_ref_words=10,
            roles=None, strip_accents=False, expand_numbers=True):
    """
    Parse the test-set JSON, normalize references, cut one wav per segment.

    min_ref_words is counted on the NORMALIZED reference, since that is what
    lands in the WER denominator: in Spanish 'son 1234 soles' is 3 raw words
    but 7 normalized ones.
    """
    norm_config = {"strip_accents": strip_accents,
                   "expand_numbers": expand_numbers}

    records, stats, failures = prepare_corpus(
        json_path, lang=lang, out_dir=out_dir, audio_root=audio_root,
        min_ref_words=min_ref_words,
        roles=set(roles) if roles else None,
        norm_config=norm_config,
    )

    report("normalization (frozen in norm_config.json)",
           sorted(norm_config.items()))
    report("parsing", sorted(stats.items()))

    if failures:
        print(f"\n{len(failures)} export failures, first 5:")
        for item in failures[:5]:
            print(f"  {item}")

    if stats["too_short"] > stats["kept"]:
        print(f"\nWARNING: {stats['too_short']} segments dropped as too short "
              f"vs {stats['kept']} kept -> lower min_ref_words if your turns "
              f"are naturally brief.")

    print("\nfirst references, raw vs normalized:")
    for record in records[:3]:
        print(f"  raw  : {record['reference']}")
        print(f"  norm : {record['reference_norm']} "
              f"({record['n_ref_words']} words)")

    return records


def sanity(records, model="nvidia/canary-180m-flash", n_per_lang=20):
    """
    Transcribe a few segments per language and print them next to the
    reference. Check that each hypothesis is in the SAME language as its
    reference before spending a GPU-hour on the full corpus.
    """
    cache = sanity_check_canary(records, model_name=model,
                                n_per_lang=n_per_lang)
    print("\nEvery hypothesis must be in the same language as its reference.")
    return cache


def transcribe(records, out_dir, systems=None):
    """
    Run the target model and the proxies over every slice.

    systems: list of roles among 'target', 'proxy_a', 'proxy_b'. Models load
    one at a time, so peak GPU memory is the largest single system, not their
    sum. Caches are resumable.
    """
    selected = SYSTEMS if not systems else {r: SYSTEMS[r] for r in systems}
    results = run_all(records, os.path.join(out_dir, "asr"), systems=selected)
    report("transcribed", [(role, f"{len(texts)} segments")
                           for role, texts in results.items()])
    return results


def accents(records, out_dir, min_delta=0.005):
    """
    Compare strip_accents on/off using the target system's own hypotheses.

    A large gap means the annotators and the ASR disagree on diacritics, which
    would otherwise be counted as recognition errors. If the verdict says so,
    rerun prepare(strip_accents=True) then merge() - no need to re-transcribe,
    the caches store raw text.
    """
    cache = load_cache(os.path.join(out_dir, "asr", "target.jsonl"))
    if not cache:
        print("No target transcriptions yet -> run transcribe(['target']).")
        return None

    by_lang = {}
    for record in records:
        row = cache.get(record["segment_id"])
        if row:
            by_lang.setdefault(record["lang"], []).append(
                (record["reference"], row["text"])
            )

    verdicts = {}
    for lang, pairs in by_lang.items():
        impact = diagnose_accent_impact(pairs, lang)
        delta = impact["keep"] - impact["strip"]
        verdicts[lang] = "strip_accents=True" if delta > min_delta else "keep as is"
        report(f"accent impact [{lang}] on {len(pairs)} pairs", [
            ("wer keeping accents", f"{impact['keep']:.4f}"),
            ("wer stripping", f"{impact['strip']:.4f}"),
            ("delta", f"{delta:.4f}"),
            ("verdict", verdicts[lang]),
        ])
    return verdicts


def merge(records, out_dir):
    """
    Join the ASR caches onto the records, normalize every hypothesis with the
    frozen config, and compute the WER labels.
    """
    asr_dir = os.path.join(out_dir, "asr")
    results = {}
    for role in SYSTEMS:
        cache = load_cache(os.path.join(asr_dir, f"{role}.jsonl"))
        if cache:
            results[role] = {sid: row["text"] for sid, row in cache.items()}

    if "target" not in results:
        print("No target transcriptions -> nothing to label.")
        return None

    rows, info = build_table(records, results, out_dir)

    report("normalization used", sorted(info["norm_config"].items()))
    report("coverage", [(role, f"{n}/{len(records)}")
                        for role, n in info["coverage"].items()])
    report("target WER on this corpus", sorted(info["summary"].items()))

    print("\nlabelled examples:")
    for row in rows[:2]:
        print(f"  ref    : {row['reference_norm']}")
        print(f"  target : {row['hyp_target_norm']}")
        print(f"  label  : {row['label_errors']} errors, "
              f"wer={row['label_wer']:.3f} "
              f"(S={row['label_S']} D={row['label_D']} I={row['label_I']})")

    print(f"\n{len(rows)} rows -> {os.path.join(out_dir, 'table.jsonl')}")
    return rows


def status(out_dir):
    """Where the corpus currently stands."""
    records = load(out_dir)
    lines = [
        ("segments", len(records)),
        ("calls", len({r["sample_id"] for r in records})),
        ("hours", round(sum(r["duration"] for r in records) / 3600.0, 3)),
        ("languages", sorted({r["lang"] for r in records})),
        ("norm_config", read_norm_config(out_dir)),
    ]
    for role in SYSTEMS:
        cache = load_cache(os.path.join(out_dir, "asr", f"{role}.jsonl"))
        lines.append((role, f"{len(cache)}/{len(records)}"))
    lines.append(("table built",
                  os.path.exists(os.path.join(out_dir, "table.jsonl"))))
    report(f"status of {out_dir}", lines)
    return dict(lines)


def to_frame(rows):
    """Merged rows as a DataFrame, for poking around in the notebook."""
    import pandas as pd

    return pd.DataFrame(rows)
