#!/usr/bin/env python3
"""
Detects the musical key of WAV files and renames them to include the key.
Example: audio.wav -> audio_Cm.wav (for C minor)

Key detection uses the Krumhansl-Schmuckler algorithm:
  Krumhansl, C.L. & Kessler, E.J. (1982). Tracing the dynamic changes in
  perceived tonal organization in a spatial representation of musical keys.
  Psychological Review, 89(4), 334-368.
"""

import os
import re
import sys
import argparse
import librosa
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Key names for the 12 chromatic pitch classes
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Kessler (1982) tonal hierarchy profiles
KEY_PROFILES = {
    'major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    'minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
}

# Regex that matches a key suffix added by this tool, e.g. _Cm or _F#
_KEY_SUFFIX_RE = re.compile(
    r'_(' + '|'.join(re.escape(k) for k in KEYS) + r')m?$'
)

# Maximum audio duration (seconds) and sample rate used for key detection.
# Full files aren't needed — 30 s at 22 kHz is sufficient for chroma analysis.
_ANALYSIS_DURATION = 30
_ANALYSIS_SR = 22050


def detect_key(audio_path: str) -> tuple[str, float]:
    """
    Detect the musical key of an audio file.

    Returns:
        (key_name, confidence) where key_name is e.g. "Cm" or "F#" and
        confidence is the Pearson r of the best-matching profile (0–1).
    """
    y, sr = librosa.load(
        audio_path,
        sr=_ANALYSIS_SR,
        duration=_ANALYSIS_DURATION,
        mono=True,
    )

    # chroma_cens is more robust to dynamics and timbre than chroma_cqt
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_vals = np.mean(chroma, axis=1)
    chroma_vals = chroma_vals / np.sum(chroma_vals)

    best_r = -1.0
    best_key = None
    best_mode = None

    for mode, raw_profile in KEY_PROFILES.items():
        profile = np.array(raw_profile)
        profile = profile / np.sum(profile)

        for i in range(12):
            rotated = np.roll(profile, i)
            r = float(np.corrcoef(chroma_vals, rotated)[0, 1])
            if r > best_r:
                best_r = r
                best_key = KEYS[i]
                best_mode = mode

    key_name = best_key + ('m' if best_mode == 'minor' else '')
    return key_name, best_r


def _already_has_key(stem: str, key: str) -> bool:
    """Return True if *stem* already ends with the key suffix we would add."""
    return bool(_KEY_SUFFIX_RE.search(stem))


def _process_one(wav_path: str, dry_run: bool) -> dict:
    """Detect key for one file and optionally rename it. Returns a result dict."""
    p = Path(wav_path)
    result = {"file": p.name, "error": None, "key": None, "confidence": None, "renamed_to": None, "skipped": False}

    try:
        key, confidence = detect_key(wav_path)
        result["key"] = key
        result["confidence"] = confidence

        if _already_has_key(p.stem, key):
            result["skipped"] = True
        else:
            new_name = f"{p.stem}_{key}{p.suffix}"
            new_path = p.parent / new_name
            result["renamed_to"] = new_name
            if not dry_run:
                p.rename(new_path)
    except Exception as exc:
        result["error"] = str(exc)

    return result


def rename_files_with_keys(directory: str = ".", dry_run: bool = False, workers: int = 4) -> None:
    """Process all WAV files in *directory* and rename them with detected keys."""
    directory = Path(directory)
    wav_files = sorted(
        set(directory.rglob("*.wav")) | set(directory.rglob("*.WAV"))
    )

    if not wav_files:
        print(f"No WAV files found in {directory}")
        return

    print(f"Found {len(wav_files)} WAV file(s)"
          + (" [DRY RUN — no files will be renamed]" if dry_run else ""))

    # Use parallel workers; fall back to serial for a single file
    futures_map = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(wav_files))) as pool:
        for wf in wav_files:
            fut = pool.submit(_process_one, str(wf), dry_run)
            futures_map[fut] = wf.name

        for fut in as_completed(futures_map):
            r = fut.result()
            _print_result(r)


def _print_result(r: dict) -> None:
    name = r["file"]
    if r["error"]:
        print(f"\n  {name}")
        print(f"    ERROR: {r['error']}")
        return

    confidence_pct = f"{r['confidence']:.0%}"
    if r["skipped"]:
        print(f"\n  {name}")
        print(f"    Key: {r['key']} ({confidence_pct} confidence) — already in filename, skipped")
    elif r["renamed_to"]:
        arrow = "would rename to" if r.get("dry_run") else "renamed to"
        print(f"\n  {name}")
        print(f"    Key: {r['key']} ({confidence_pct} confidence) → {r['renamed_to']}")
    else:
        print(f"\n  {name}")
        print(f"    Key: {r['key']} ({confidence_pct} confidence) → {r['renamed_to'] or 'renamed'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect musical keys of WAV files and rename them."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without changing any files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.directory}\n")
    rename_files_with_keys(args.directory, dry_run=args.dry_run, workers=args.workers)
    print("\nDone!")


if __name__ == "__main__":
    main()
