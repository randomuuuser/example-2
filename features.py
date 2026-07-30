"""
Block 4a - Features.

Two blocks are built here, in the order the ablation of Waheed et al. (ACL
Findings 2025) says they matter:

  proxy   the agreement between the target hypothesis and each proxy
          hypothesis, used as a pseudo-reference. Removing it tripled their
          MAE (1.03 -> 3.13), so this is the load-bearing block.
  text    cheap surface statistics of the target hypothesis alone.

A third block (decoder logprobs / entropies) will slot in later; every
function here takes the row dict, so adding columns costs nothing.

Naming convention: everything is computed on the *_norm fields, which went
through the frozen normalization config. Never mix raw and normalized text in
the same comparison.
"""

import gzip
import math
from collections import Counter

from text_norm import wer_counts

PROXY_ROLES = ("proxy_a", "proxy_b")


def _char_error_rate(reference, hypothesis):
    """CER between two normalized strings, via the word-level machinery."""
    ref_chars = " ".join(reference.replace(" ", ""))
    hyp_chars = " ".join(hypothesis.replace(" ", ""))
    counts = wer_counts(ref_chars, hyp_chars)
    return counts["wer"]


def proxy_features(row, roles=PROXY_ROLES):
    """
    Agreement between the target hypothesis and each proxy.

    The proxy hypothesis plays the role of the reference, so its length is the
    denominator - exactly what happens at inference time, where the human
    reference does not exist.

    Also computes the proxy-vs-proxy disagreement, which does not involve the
    target at all and therefore measures how hard the segment is for everyone.
    That separates 'the target slipped here' from 'nobody gets this segment'.
    """
    target = row.get("hyp_target_norm") or ""
    features = {}
    per_role = {}

    for role in roles:
        proxy = row.get(f"hyp_{role}_norm")
        if proxy is None:
            continue
        counts = wer_counts(proxy, target)
        n_ref = max(counts["n_ref"], 1)
        per_role[role] = counts["wer"]
        features.update({
            f"pwer_{role}": counts["wer"],
            f"pcer_{role}": _char_error_rate(proxy, target),
            f"psub_{role}": counts["S"] / n_ref,
            f"pdel_{role}": counts["D"] / n_ref,
            f"pins_{role}": counts["I"] / n_ref,
            f"plen_ratio_{role}": counts["n_hyp"] / n_ref,
        })

    if per_role:
        values = list(per_role.values())
        features["pwer_mean"] = sum(values) / len(values)
        features["pwer_min"] = min(values)
        features["pwer_max"] = max(values)
        features["pwer_spread"] = max(values) - min(values)

    # Proxy-vs-proxy: intrinsic segment difficulty, target-independent
    if len(roles) >= 2:
        first = row.get(f"hyp_{roles[0]}_norm")
        second = row.get(f"hyp_{roles[1]}_norm")
        if first is not None and second is not None:
            features["pwer_proxy_vs_proxy"] = wer_counts(first, second)["wer"]

    return features


def text_features(row):
    """
    Surface statistics of the target hypothesis alone.

    Cheap, always available, and they catch the failure modes that agreement
    features miss: insertion loops inflate words_per_second and drop the gzip
    ratio, truncations do the opposite.
    """
    text = row.get("hyp_target_norm") or ""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    duration = max(float(row.get("duration") or 0.0), 1e-6)

    encoded = text.encode("utf-8")
    compressed = len(gzip.compress(encoded)) if encoded else 0
    gzip_ratio = compressed / len(encoded) if encoded else 0.0

    counts = Counter(words)
    repeated = sum(c for c in counts.values() if c > 1)

    return {
        "n_hyp_words": n_words,
        "n_hyp_chars": n_chars,
        "duration": duration,
        "words_per_second": n_words / duration,
        "chars_per_word": n_chars / n_words if n_words else 0.0,
        "gzip_ratio": gzip_ratio,
        "type_token_ratio": len(counts) / n_words if n_words else 0.0,
        "repeated_word_ratio": repeated / n_words if n_words else 0.0,
        "max_word_repeat": max(counts.values()) if counts else 0,
        "log_duration": math.log1p(duration),
    }


FEATURE_BLOCKS = {
    "proxy": proxy_features,
    "text": text_features,
}


def build_features(rows, blocks=("proxy", "text"), roles=PROXY_ROLES):
    """
    Turn merged rows into (X, y_errors, y_wer, groups, feature_names).

    y_errors is what the regressor predicts, following Waheed et al.: absolute
    error counts rather than the rate. The WER is recovered by dividing by the
    hypothesis length, which stays available without a reference.

    groups is the call id - every split must be grouped by call, because
    segments from one call share speaker, channel and line quality.
    """
    matrix, y_errors, y_wer, groups = [], [], [], []

    for row in rows:
        features = {}
        for name in blocks:
            builder = FEATURE_BLOCKS[name]
            features.update(
                builder(row, roles) if name == "proxy" else builder(row)
            )
        matrix.append(features)
        y_errors.append(float(row["label_errors"]))
        y_wer.append(float(row["label_wer"]))
        groups.append(row["sample_id"])

    names = sorted({key for features in matrix for key in features})
    X = [[features.get(name, 0.0) for name in names] for features in matrix]
    return X, y_errors, y_wer, groups, names


def block_of(name):
    """Which block a feature name belongs to, for the ablation."""
    return "proxy" if name.startswith(("pwer", "pcer", "psub", "pdel",
                                       "pins", "plen")) else "text"
