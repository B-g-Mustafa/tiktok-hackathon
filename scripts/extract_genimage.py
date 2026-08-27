#!/usr/bin/env python3
"""Extract GenImage's split zip archives correctly and without extra disk usage.

The bug you hit
----------------
Each GenImage category is ONE zip archive split into volumes:

    imagenet_ai_0508_adm.z01, .z02, ..., .z12, .zip

`.zip` is the LAST volume -- it holds the archive's central directory (the
index of every file's location), not a self-contained archive. Running plain
`unzip imagenet_ai_0508_adm.zip` reads that index, but the index points to
byte offsets that span across the OTHER files (.z01, .z02, ...) which unzip
never opens. It seeks to those offsets inside just the one file it did open,
lands on garbage, and (correctly, if confusingly) reports it as a corrupt /
zip-bomb-shaped file. Reproduced and confirmed locally before writing this:
plain `unzip` on a real multi-volume test archive produced the exact same
"bad zipfile offset" error, extracting 0 files.

The fix, and why it avoids extra disk
--------------------------------------
`7z` (p7zip) understands multi-volume zip archives natively: point it at the
`.zip` file with the `.z01`..`.zNN` siblings in the same directory, and it
reads across all of them and extracts directly -- no intermediate combined
copy is ever created. Verified locally: after extraction, the source
directory's size was UNCHANGED (still just the original volumes), and the
output was byte-identical to the original files.

If `7z` isn't available and you can't install it, the correct fallback is
`zip -s 0 file.zip --out combined.zip` (NOT `cat z01 z02 ... zip > out.zip` --
that was tried and tested here too, and a naive glob-based concatenation
order is easy to get subtly wrong, corrupting the join in exactly the way that
originally motivated an "am I sure this is right" script instead of a
one-liner). The `zip -s0` join needs one full extra copy of that CATEGORY's
data temporarily (not the whole dataset) -- this script deletes the combined
copy immediately after a verified extraction, and processes one category at a
time, so the extra usage never exceeds one category's size.

Usage
-----
    # Check what would happen, without extracting anything
    python scripts/extract_genimage.py --root /path/to/gen-image-dataset --dry-run

    # Extract everything found under --root
    python scripts/extract_genimage.py --root /path/to/gen-image-dataset

    # Just one category, and reclaim disk by deleting the zip volumes
    # afterwards (only after a verified-successful extraction)
    python scripts/extract_genimage.py --root /path/to/gen-image-dataset \
        --categories ADM --delete-originals-after-verify
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("extract_genimage")

# Matches "imagenet_ai_0508_adm.z01", "...z12", "...zip" etc.
VOLUME_RE = re.compile(r"^(?P<stem>.+)\.(?:z(?P<num>\d{2,3})|zip)$", re.IGNORECASE)

# GenImage's expected internal layout once extracted, used for a sanity check.
EXPECTED_SUBDIRS = ("train/ai", "train/nature", "val/ai", "val/nature")


def find_7z() -> str | None:
    for candidate in ("7z", "7za", "7zz"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def find_archive_groups(category_dir: Path) -> dict[str, list[Path]]:
    """Group volume files by archive stem.

    Normally one archive per category, but this doesn't assume that -- it
    groups whatever `.zNN`/`.zip` files it finds by their shared prefix.
    """
    groups: dict[str, list[Path]] = {}
    for path in category_dir.iterdir():
        if not path.is_file():
            continue
        match = VOLUME_RE.match(path.name)
        if match:
            groups.setdefault(match.group("stem"), []).append(path)
    return groups


def check_volume_completeness(stem: str, volumes: list[Path]) -> list[str]:
    """Look for gaps or a missing final .zip -- the OTHER common cause of
    exactly this error (a truncated/interrupted download, not a tool
    mismatch). Returns a list of problems; empty means it looks complete.
    """
    problems = []
    numbered = []
    has_zip = False

    for path in volumes:
        match = VOLUME_RE.match(path.name)
        if match.group("num"):
            numbered.append(int(match.group("num")))
        else:
            has_zip = True

    if not has_zip:
        problems.append(
            f"{stem}: no final .zip volume found (only numbered .zNN parts) -- "
            f"the archive's central directory is missing, download is incomplete"
        )

    if numbered:
        numbered.sort()
        expected = list(range(1, numbered[-1] + 1))
        missing = sorted(set(expected) - set(numbered))
        if missing:
            width = len(re.match(VOLUME_RE, volumes[0].name).group("num"))
            missing_names = [f".z{n:0{width}d}" for n in missing]
            problems.append(f"{stem}: missing volume(s) {missing_names}")

    return problems


def estimate_uncompressed_size(zip_path: Path, sevenzip_bin: str) -> int | None:
    """Ask 7z for the archive's total uncompressed size, without extracting."""
    try:
        result = subprocess.run(
            [sevenzip_bin, "l", str(zip_path)],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    for line in result.stdout.splitlines():
        if line.strip().lower().startswith("size:") or "files," in line.lower():
            match = re.search(r"(\d+)\s+\d+\s+\d+\s+files", line)
            if match:
                return int(match.group(1))
    # Fall back to scanning the summary line format 7z actually prints:
    # "Size:       123456789"
    for line in result.stdout.splitlines():
        match = re.match(r"\s*Size:\s+(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def verify_extracted(output_dir: Path) -> list[str]:
    """Check the extracted tree actually looks like GenImage. Missing pieces
    here mean don't trust the extraction enough to delete the source."""
    problems = []
    for subdir in EXPECTED_SUBDIRS:
        target = output_dir / subdir
        if not target.is_dir():
            problems.append(f"missing expected directory: {subdir}")
            continue
        if not any(target.iterdir()):
            problems.append(f"expected directory is empty: {subdir}")
    return problems


def extract_with_7z(zip_path: Path, output_dir: Path, sevenzip_bin: str) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sevenzip_bin, "x", str(zip_path), f"-o{output_dir}", "-y"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("7z failed on %s:\n%s", zip_path, result.stdout + result.stderr)
        return False
    return True


def extract_with_zip_join(zip_path: Path, output_dir: Path) -> bool:
    """Fallback when 7z is unavailable: join with `zip -s0`, extract, then
    delete the joined copy immediately. Bounded extra disk = one archive's
    worth, for as long as this function runs -- not the whole dataset, and
    not left behind afterwards.
    """
    if shutil.which("zip") is None or shutil.which("unzip") is None:
        logger.error("neither 7z nor zip/unzip found -- cannot extract")
        return False

    joined = zip_path.with_name(f"_joined_{zip_path.stem}.zip")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        join_result = subprocess.run(
            ["zip", "-s", "0", str(zip_path), "--out", str(joined)],
            capture_output=True, text=True,
        )
        if join_result.returncode != 0:
            logger.error(
                "zip -s0 join failed for %s:\n%s",
                zip_path, join_result.stdout + join_result.stderr,
            )
            return False

        extract_result = subprocess.run(
            ["unzip", "-q", "-o", str(joined), "-d", str(output_dir)],
            capture_output=True, text=True,
        )
        if extract_result.returncode != 0:
            logger.error(
                "unzip failed on joined archive for %s:\n%s",
                zip_path, extract_result.stdout + extract_result.stderr,
            )
            return False
    finally:
        # Always clean up the joined copy -- this is the temporary disk cost,
        # and it must not outlive this single archive's extraction.
        joined.unlink(missing_ok=True)

    return True


def process_category(
    category_dir: Path,
    sevenzip_bin: str | None,
    dry_run: bool,
    delete_originals: bool,
) -> bool:
    groups = find_archive_groups(category_dir)
    if not groups:
        logger.warning("%s: no .zNN/.zip volumes found, skipping", category_dir.name)
        return True

    all_ok = True

    for stem, volumes in groups.items():
        problems = check_volume_completeness(stem, volumes)
        if problems:
            for problem in problems:
                logger.error("  %s", problem)
            logger.error(
                "%s: incomplete download -- re-run the download script for "
                "this category rather than extracting a partial archive",
                stem,
            )
            all_ok = False
            continue

        zip_path = next(v for v in volumes if v.suffix.lower() == ".zip")
        output_dir = category_dir / "extracted"

        total_volume_bytes = sum(v.stat().st_size for v in volumes)
        free_bytes = shutil.disk_usage(category_dir).free

        logger.info(
            "%s: %d volumes, %.2f GB compressed, %.2f GB free on this filesystem",
            stem, len(volumes), total_volume_bytes / 1e9, free_bytes / 1e9,
        )

        if sevenzip_bin and free_bytes < total_volume_bytes * 1.05:
            logger.warning(
                "  free space is close to the archive size -- extraction needs "
                "roughly 1x the uncompressed size (images don't compress much "
                "further once already JPEG/PNG, so uncompressed ~= compressed here)"
            )

        if dry_run:
            logger.info("  [dry-run] would extract -> %s", output_dir)
            continue

        if sevenzip_bin:
            logger.info("  extracting with 7z (no intermediate copy) ...")
            success = extract_with_7z(zip_path, output_dir, sevenzip_bin)
        else:
            logger.info(
                "  7z not found -- falling back to zip -s0 join "
                "(temporary extra ~%.2f GB, deleted immediately after)",
                total_volume_bytes / 1e9,
            )
            success = extract_with_zip_join(zip_path, output_dir)

        if not success:
            all_ok = False
            continue

        verify_problems = verify_extracted(output_dir)
        if verify_problems:
            logger.error("%s: extraction finished but looks wrong:", stem)
            for problem in verify_problems:
                logger.error("  %s", problem)
            all_ok = False
            continue

        logger.info("  OK: %s", output_dir)

        if delete_originals:
            logger.info("  deleting %d source volume(s) for %s", len(volumes), stem)
            for volume in volumes:
                volume.unlink()

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Directory containing category subfolders (ADM, BIGGAN, Glide, ...).",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Only these category subfolder names (default: all found under --root).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check volumes and disk space, report the plan, extract nothing.",
    )
    parser.add_argument(
        "--delete-originals-after-verify", action="store_true", dest="delete_originals",
        help="After a category extracts AND passes the structure check, delete "
             "its .zNN/.zip volumes to reclaim space. Off by default -- this is "
             "a permanent, irreversible deletion of your only local copy.",
    )
    args = parser.parse_args()

    configure_logging()

    if not args.root.is_dir():
        logger.error("not a directory: %s", args.root)
        return 2

    sevenzip_bin = find_7z()
    if sevenzip_bin:
        logger.info("using 7z: %s", sevenzip_bin)
    else:
        logger.warning(
            "7z not found on PATH -- will fall back to zip -s0 join per "
            "category (works, but needs temporary disk equal to one "
            "category's size). Install p7zip if you can (conda install -c "
            "conda-forge p7zip, or `module load` it if your cluster provides "
            "one) to avoid that entirely."
        )

    if args.delete_originals:
        logger.warning(
            "--delete-originals-after-verify is ON: source .zNN/.zip volumes "
            "will be PERMANENTLY DELETED after each category's extraction is "
            "verified. There is no undo -- you would need to re-download."
        )

    category_dirs = (
        [args.root / c for c in args.categories]
        if args.categories
        else sorted(p for p in args.root.iterdir() if p.is_dir())
    )

    overall_ok = True
    for category_dir in category_dirs:
        if not category_dir.is_dir():
            logger.error("no such category directory: %s", category_dir)
            overall_ok = False
            continue

        logger.info("\n" + "=" * 72)
        logger.info("CATEGORY: %s", category_dir.name)
        logger.info("=" * 72)

        ok = process_category(
            category_dir, sevenzip_bin, args.dry_run, args.delete_originals
        )
        overall_ok = overall_ok and ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
