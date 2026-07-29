"""
Block 1 - Unified text normalization for WER / pWER computation.

Every hypothesis (target model + proxies) AND the human reference must go
through the exact same function. Any asymmetry here turns a convention
mismatch into a fake recognition error, which corrupts both the labels and
the dominant feature (pWER).

Supported languages: 'es', 'de' (extendable via NUM2WORDS_LANG).
"""

import re
import unicodedata

from num2words import num2words

# num2words language codes, keyed by our corpus language code
NUM2WORDS_LANG = {"es": "es", "de": "de", "en": "en", "fr": "fr"}

# Characters we keep: letters (incl. accents/umlauts/eszett), digits, spaces,
# and the apostrophe (meaningful in some languages). Everything else is dropped.
_KEEP_APOSTROPHE = "'"

# A number token: optional sign, digit groups possibly separated by . or ,
# Examples matched: 5  1234  1.234  1,234  3,5  12.345,67  -7
_NUMBER_RE = re.compile(r"[-+]?\d[\d.,]*")


# German systems disagree on umlaut spelling ('möchte' vs 'moechte'), so the
# fold must TRANSLITERATE (oe) rather than strip (o) to unify both forms.
_GERMAN_FOLD = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def _strip_accents(text, lang=None):
    """
    Fold diacritics to a single canonical form.

    German: transliterate umlauts to digraphs, so 'möchte' and 'moechte' meet.
    Other languages: drop the diacritic and keep the base letter (á -> a).
    """
    if lang == "de":
        for source, target in _GERMAN_FOLD.items():
            text = text.replace(source, target)
    else:
        text = text.replace("ß", "ss")

    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _parse_number(raw):
    """
    Turn a raw digit string into a float or int, or None if ambiguous.

    Handles both '1.234,56' (de/es style) and '1,234.56' (en style) by
    treating the LAST separator as the decimal mark only when it is followed
    by 1-2 digits; otherwise all separators are thousand marks.
    """
    sign = -1 if raw.startswith("-") else 1
    body = raw.lstrip("+-").rstrip(".,")
    if not body:
        return None

    seps = [c for c in body if c in ".,"]
    if not seps:
        return sign * int(body)

    last_sep = body.rfind(seps[-1])
    tail = body[last_sep + 1:]

    # A 3-digit tail is a thousand group ('1.234'); a 1-2 digit tail is a
    # decimal part ('3,5' / '12.345,67').
    if len(tail) in (1, 2):
        integer_part = re.sub(r"[.,]", "", body[:last_sep])
        integer_part = integer_part or "0"
        return sign * float(f"{integer_part}.{tail}")

    return sign * int(re.sub(r"[.,]", "", body))


def _expand_numbers(text, lang):
    """
    Replace every digit string by its spelled-out form.

    This makes '25' and 'veinticinco' comparable. Direction matters little as
    long as it is the same for every system; spelling out is the safer way
    because word-to-digit parsing is far more error-prone in es/de.
    """
    n2w_lang = NUM2WORDS_LANG.get(lang, "en")

    def replace(match):
        value = _parse_number(match.group(0))
        if value is None:
            return " "
        try:
            return " " + num2words(value, lang=n2w_lang) + " "
        except (NotImplementedError, OverflowError, ValueError):
            # Fall back to reading digit by digit
            digits = re.sub(r"\D", "", match.group(0))
            return " " + " ".join(num2words(int(d), lang=n2w_lang) for d in digits) + " "

    return _NUMBER_RE.sub(replace, text)


def normalize_text(text, lang, strip_accents=False, expand_numbers=True):
    """
    Normalize one transcript for WER / pWER computation.

    Args:
        text: raw hypothesis or reference.
        lang: 'es', 'de', ...
        strip_accents: merge accented and unaccented forms. Reduces spurious
            differences between systems that disagree on diacritics, at the
            cost of merging genuinely distinct German words (schon/schoen).
            Measure the
            impact with diagnose_accent_impact before turning this on.
        expand_numbers: spell out every digit string.

    Returns:
        A lowercase, punctuation-free, single-spaced string.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    if expand_numbers:
        text = _expand_numbers(text, lang)

    if strip_accents:
        text = _strip_accents(text, lang)

    # Drop everything that is not a letter, a digit or an apostrophe
    text = "".join(
        c if (c.isalnum() or c == _KEEP_APOSTROPHE) else " " for c in text
    )

    return " ".join(text.split())


def normalize_all(records, lang_key="lang", text_keys=(), **norm_kwargs):
    """
    Apply normalize_text in place over a list of dicts (one per segment).

    Every text field listed in text_keys gets a '<key>_norm' twin, so the raw
    strings stay available for manual inspection.
    """
    for record in records:
        lang = record[lang_key]
        for key in text_keys:
            record[f"{key}_norm"] = normalize_text(record.get(key), lang, **norm_kwargs)
    return records


def wer_counts(reference, hypothesis):
    """
    Edit counts and WER between two already-normalized strings.

    Used both for the target label (vs human reference) and for the proxy
    features (vs proxy hypothesis). Returns zeros-safe values on empty input.
    """
    import jiwer

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        n_ins = len(hyp_words)
        return {"S": 0, "D": 0, "I": n_ins, "n_ref": 0, "n_hyp": n_ins,
                "errors": n_ins, "wer": float(n_ins > 0)}

    out = jiwer.process_words(reference, hypothesis)
    errors = out.substitutions + out.deletions + out.insertions
    return {
        "S": out.substitutions,
        "D": out.deletions,
        "I": out.insertions,
        "n_ref": len(ref_words),
        "n_hyp": len(hyp_words),
        "errors": errors,
        "wer": errors / len(ref_words),
    }


def diagnose_accent_impact(pairs, lang):
    """
    Measure how much the strip_accents choice moves the WER.

    pairs: list of (reference, hypothesis) raw strings.
    Run this once per corpus before freezing the normalization config. A large
    gap means the systems disagree on diacritics and stripping is worth it.
    """
    totals = {"keep": [0, 0], "strip": [0, 0]}
    for reference, hypothesis in pairs:
        for mode, strip in (("keep", False), ("strip", True)):
            ref = normalize_text(reference, lang, strip_accents=strip)
            hyp = normalize_text(hypothesis, lang, strip_accents=strip)
            counts = wer_counts(ref, hyp)
            totals[mode][0] += counts["errors"]
            totals[mode][1] += counts["n_ref"]

    return {
        mode: (errs / n_ref if n_ref else 0.0)
        for mode, (errs, n_ref) in totals.items()
    }


if __name__ == "__main__":
    samples = [
        ("es", "Sí, son 25 euros con IVA.", "si son veinticinco euros con iva"),
        ("es", "El total es 1.234,50 soles.", "el total es mil doscientos treinta y cuatro coma cinco soles"),
        ("de", "Das kostet 25 Euro, oder?", "Das kostet fünfundzwanzig Euro oder"),
        ("de", "Ich rufe am 3. Mai an.", "ich rufe am dritten mai an"),
    ]
    for lang, a, b in samples:
        na, nb = normalize_text(a, lang), normalize_text(b, lang)
        print(f"[{lang}] {na}\n[{lang}] {nb}\n      wer={wer_counts(na, nb)['wer']:.3f}\n")
