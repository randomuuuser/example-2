"""Filename validation: whitelist-based, ASCII printable + accented letters."""

import string
import unicodedata

# The whitelist IS the rule. Everything not listed here is rejected.
SAFE_PUNCTUATION = "._- ()"
SAFE_ASCII = frozenset(string.ascii_letters + string.digits + SAFE_PUNCTUATION)

# Accepted beyond ASCII: letters and combining accents only. No symbols (So,
# Sm, Sc, Sk) -> no emoji, no currency or math signs.
LETTER_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc"})

# Unicode categories that never render as real text. Used for diagnosis only.
UNRENDERABLE = {
    "Cc": "control character",
    "Cf": "invisible formatting character",
    "Cs": "unpaired surrogate",
    "Co": "private use character (renders as an empty box)",
    "Cn": "unassigned code point (renders as an empty box)",
}

# Also diagnosis only: these are already excluded by the whitelist.
FILESYSTEM_RESERVED = frozenset('<>:"/\\|?*')

WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
}


def _reject_reason(char, allowed):
    """Return why char is rejected, or None if it is acceptable."""
    if char in allowed:
        return None  # fast path: no unicodedata lookup for plain ASCII

    category = unicodedata.category(char)
    if category in UNRENDERABLE:
        return UNRENDERABLE[category]
    if char in FILESYSTEM_RESERVED:
        return "character reserved by the filesystem"
    if char.isascii():
        return "ASCII character unsafe in shells, URLs or scripts"
    if category in LETTER_CATEGORIES:
        return None
    return f"non-ASCII symbol (category {category})"


def check_filename(filename, max_length=255, extra_allowed=""):
    """Return (is_valid, reason). Reason is empty when the name is valid."""
    if not isinstance(filename, str):
        return False, f"Filename must be a string, got {type(filename).__name__}."

    if not filename.strip():
        return False, "Filename is empty or contains only whitespace."

    if len(filename) > max_length:
        return False, f"Filename is {len(filename)} characters long, max is {max_length}."

    allowed = SAFE_ASCII.union(extra_allowed) if extra_allowed else SAFE_ASCII

    for position, char in enumerate(filename):
        reason = _reject_reason(char, allowed)
        if reason:
            return False, (
                f"Invalid character {char!r} (U+{ord(char):04X}) "
                f"at position {position}: {reason}."
            )

    if filename[0] == " " or filename[-1] in " .":
        return False, "Filename must not start with a space or end with a space or a dot."

    stem = filename.split(".", 1)[0].strip().upper()
    if stem in WINDOWS_RESERVED:
        return False, f"{stem!r} is a reserved device name on Windows."

    return True, ""


# def find_all_invalid(filename, extra_allowed=""):
#     """Return every rejected character as a list of (position, char, reason)."""
#     allowed = SAFE_ASCII.union(extra_allowed) if extra_allowed else SAFE_ASCII
#     issues = []
#     for position, char in enumerate(filename):
#         reason = _reject_reason(char, allowed)
#         if reason:
#             issues.append((position, char, f"U+{ord(char):04X}: {reason}"))
#     return issues


if __name__ == "__main__":
    samples = [
        "rapport_2026.pdf",
        "compte rendu (v2).wav",
        "résumé_réunion.docx",
        "re\u0301union_nfd.docx",   # decomposed accent, macOS style
        "compte\uf03arendu.wav",    # private use area -> empty box
        "audio\u200bfile.mp3",      # zero width space
        "call*record.wav",
        "note~1.txt",
        "meeting_\U0001f4de.txt",   # emoji
        "prix_\u20ac100.csv",       # currency symbol
        "NUL.txt",
        "trailing_dot.",
        "\u00e9" * 200 + ".txt",    # 200 chars but 400+ bytes
    ]
    for name in samples:
        valid, reason = check_filename(name)
        label = name if len(name) <= 28 else name[:25] + "..."
        print(f"{valid!s:<5} {label!r:<32} {reason}")
