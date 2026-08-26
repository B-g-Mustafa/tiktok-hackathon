#!/usr/bin/env python3
"""
Download a size-limited subset of the GenImage dataset from
Google Drive.

Each GenImage category is one zip archive split across many Drive
files (foo.z01, foo.z02, ..., foo.zip). All volumes are required to
extract anything, so for each category this script downloads either
ALL of its volumes (if the category's total size is <= --max-gb) or
none at all (category is skipped).

Dependencies (install on the machine that runs this script):

    pip install gdown requests
"""

import argparse
import re
import sys
import time
from pathlib import Path

try:
    import gdown
    import requests
except ImportError as exc:
    print(f"ERROR: missing dependency ({exc}).")
    print("Install with: pip install gdown requests")
    sys.exit(1)


GENIMAGE_URL = (
    "https://drive.google.com/drive/folders/"
    "1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
)

# The eight original GenImage generator categories.
CATEGORIES = {
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
}


def get_drive_listing(folder_url: str):
    """
    Recursively list every file in the Drive folder (id + relative
    path) WITHOUT downloading any file contents.
    """

    print("Reading Google Drive folder structure...")

    try:
        entries = gdown.download_folder(
            url=folder_url,
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )
    except Exception as exc:
        print("ERROR: Could not list Google Drive folder.")
        print(exc)
        sys.exit(1)

    if not entries:
        print("ERROR: Google Drive folder listing came back empty.")
        sys.exit(1)

    files = []

    for entry in entries:
        file_id = getattr(entry, "id", None)
        path = getattr(entry, "path", None) or getattr(
            entry, "local_path", None
        )

        if file_id is None and isinstance(entry, dict):
            file_id = entry.get("id")
            path = entry.get("path") or entry.get("local_path")

        if file_id is None or path is None:
            continue

        files.append({"id": file_id, "path": str(path)})

    return files


def get_category(path: str):
    """
    Extract the GenImage generator category from a Drive path.

    Example:
        GenImage/ADM/foo.zip
        -> ADM
    """

    parts = Path(path).parts

    for part in parts:
        if part in CATEGORIES:
            return part

    # Case-insensitive fallback
    for part in parts:
        for category in CATEGORIES:
            if part.lower() == category.lower():
                return category

    return None


# Each GenImage category is ONE zip archive split across many Drive
# files: foo.z01, foo.z02, ..., foo.zip. The trailing ".zip" file is
# the LAST volume (it holds the central directory), not a separate
# archive. Every volume is required to extract anything, so these
# must always be downloaded as a complete set.
ARCHIVE_PART_RE = re.compile(r"\.(zip|z\d{2,3})$", re.IGNORECASE)


def is_archive_part(path: str) -> bool:
    return bool(ARCHIVE_PART_RE.search(path))


def part_sort_key(path: str):
    """
    Sort split-archive volumes in download order: z01, z02, ...,
    then the final .zip volume last (it must be written last since
    it's what most tools check for when reassembling the archive).
    """

    ext = Path(path).suffix.lower()

    if ext == ".zip":
        return (1, 0)

    match = re.match(r"\.z(\d+)$", ext)

    if match:
        return (0, int(match.group(1)))

    return (2, ext)


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def _extract_confirm_params(html_text: str):
    """
    Parse Google Drive's "can't scan this file for viruses"
    interstitial page for the hidden form fields needed to
    actually reach the file (confirm token / uuid).
    """

    params = {}

    confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html_text)
    uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html_text)

    if confirm:
        params["confirm"] = confirm.group(1)

    if uuid:
        params["uuid"] = uuid.group(1)

    return params


def probe_file_size(session: "requests.Session", file_id: str):
    """
    Best-effort lookup of a Drive file's size WITHOUT downloading
    its contents, so the per-category GB budget can be checked
    *before* starting a multi-GB download.

    Returns the size in bytes, or None if it could not be
    determined (the caller then falls back to counting the actual
    size after the file has been downloaded).
    """

    base_url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}

    try:
        response = session.get(
            base_url,
            params=params,
            stream=True,
            timeout=30,
        )
    except requests.RequestException:
        return None

    content_type = response.headers.get("Content-Type", "")

    # Small files: Drive serves the file directly, no interstitial.
    if "text/html" not in content_type:
        size = response.headers.get("Content-Length")
        response.close()
        return int(size) if size is not None else None

    # Large files: Drive shows a virus-scan warning page first.
    html_text = response.text
    response.close()

    confirm_params = _extract_confirm_params(html_text)

    if not confirm_params:
        # Older Drive flow: confirm token comes back as a cookie.
        for key, value in session.cookies.items():
            if key.startswith("download_warning"):
                confirm_params = {"confirm": value}
                break

    if not confirm_params:
        return None

    params.update(confirm_params)

    try:
        response = session.get(
            "https://drive.usercontent.google.com/download",
            params=params,
            stream=True,
            timeout=30,
        )
    except requests.RequestException:
        return None

    size = response.headers.get("Content-Length")
    response.close()

    if size is not None:
        return int(size)

    # Last resort: the warning text itself often states the size,
    # e.g. "... (3.5G) is too large for Google to scan ...".
    match = re.search(r"\(([\d.]+)\s*([KMGT])\)", html_text)

    if not match:
        return None

    value, unit = match.groups()
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[unit]

    return int(float(value) * multiplier)


def estimate_category_size(session: "requests.Session", files):
    """
    Estimate a category's total archive size WITHOUT probing every
    part. Split-archive volumes are all the same size except the
    last one (guaranteed by how `zip -s` splitting works), so we
    only need to probe the first and last volume.

    Returns bytes, or None if it could not be determined.
    """

    if len(files) == 1:
        return probe_file_size(session, files[0]["id"])

    first_size = probe_file_size(session, files[0]["id"])
    last_size = probe_file_size(session, files[-1]["id"])

    if first_size is None or last_size is None:
        return None

    return first_size * (len(files) - 1) + last_size


def download_file(
    file_id: str,
    dest_path: Path,
    retries: int = 3,
    backoff_seconds: int = 15,
) -> bool:
    """
    Download one Google Drive file (by id) with resume support.

    Retries on failure (e.g. Drive's transient "too many users have
    downloaded this file recently" quota errors), since a single
    missing volume makes the whole split archive unusable.
    """

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):

        print(
            f"\nDownloading (attempt {attempt}/{retries}):\n"
            f"  id={file_id}\n"
            f"  -> {dest_path}"
        )

        try:
            result = gdown.download(
                id=file_id,
                output=str(dest_path),
                quiet=False,
                resume=True,
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            result = None

        if result is not None:
            return True

        if attempt < retries:
            wait = backoff_seconds * attempt
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    return False


def download_genimage(
    output_dir: Path,
    max_gb: float,
):
    """
    For EACH GenImage generator category, download the category's
    complete split-archive (all its .z01/.z02/.../.zip volumes)
    ONLY IF the category's total size fits within max_gb.
    Categories that exceed max_gb are skipped entirely, since a
    partial set of volumes can't be extracted anyway.
    """

    max_bytes = int(max_gb * 1024 ** 3)

    entries = get_drive_listing(GENIMAGE_URL)

    # ---------------------------------------------------------
    # Organize archive volumes by generator
    # ---------------------------------------------------------

    categories = {
        category: []
        for category in CATEGORIES
    }

    for entry in entries:

        path = entry["path"]

        if not is_archive_part(path):
            continue

        category = get_category(path)

        if category is None:
            continue

        categories[category].append(
            {
                "path": path,
                "id": entry["id"],
            }
        )

    for category in categories:
        categories[category].sort(
            key=lambda x: part_sort_key(x["path"])
        )

    # ---------------------------------------------------------
    # Show the plan
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("GenImage Download Plan")
    print(f"Per-category limit: {max_gb:.2f} GB")
    print("=" * 80)

    for category in sorted(categories):

        files = categories[category]

        print(
            f"\n{category}: "
            f"{len(files)} archive volumes"
        )

    print("=" * 80)

    # ---------------------------------------------------------
    # Process categories
    # ---------------------------------------------------------

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for category in sorted(categories):

        files = categories[category]

        print("\n")
        print("=" * 80)
        print(f"CATEGORY: {category}")
        print("=" * 80)

        if not files:
            print(f"[WARNING] No archive volumes found for {category}")
            continue

        estimated_size = estimate_category_size(session, files)

        if estimated_size is None:
            print(
                f"[SKIP] Could not determine {category}'s size "
                f"(Drive request failed) -- skipping to avoid "
                f"downloading an unusable partial archive."
            )
            continue

        print(
            f"Estimated size: {human_size(estimated_size)} "
            f"({len(files)} volumes) -- limit {max_gb:.2f} GB"
        )

        if estimated_size > max_bytes:
            print(
                f"[SKIP] {category} exceeds the {max_gb:.2f} GB "
                f"per-category limit."
            )
            continue

        category_dir = output_dir / category

        downloaded_bytes = 0
        downloaded_files = 0
        incomplete = False

        for file_info in files:

            path = file_info["path"]
            file_id = file_info["id"]
            dest_path = category_dir / Path(path).name

            print(
                f"\n[{downloaded_files + 1}/{len(files)}] "
                f"{Path(path).name}"
            )

            success = download_file(
                file_id=file_id,
                dest_path=dest_path,
            )

            if not success:

                print(
                    f"[ERROR] Failed to download {path} "
                    f"after retries."
                )

                print(
                    f"[INCOMPLETE] {category} is missing volume(s); "
                    f"rerun the script later to retry -- gdown will "
                    f"resume any partial file and skip files it "
                    f"already has."
                )

                incomplete = True
                break

            if dest_path.exists():
                downloaded_bytes += dest_path.stat().st_size

            downloaded_files += 1

            print(
                f"Category total: "
                f"{human_size(downloaded_bytes)} "
                f"({downloaded_files}/{len(files)} volumes)"
            )

        status = "INCOMPLETE" if incomplete else "COMPLETE"

        print(f"\n[{status}] {category}")
        print(f"Volumes downloaded: {downloaded_files}/{len(files)}")
        print(f"Total: {human_size(downloaded_bytes)}")

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    for category in sorted(categories):

        category_dir = (
            output_dir / category
        )

        if not category_dir.exists():
            continue

        files = [
            f for f in category_dir.iterdir()
            if f.is_file()
        ]

        total = sum(
            f.stat().st_size
            for f in files
        )

        print(
            f"{category:<30} "
            f"{len(files):>3} volumes  "
            f"{human_size(total)}"
        )

    print("=" * 80)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Download a size-limited subset of "
            "the original GenImage dataset."
        )
    )

    parser.add_argument(
        "--max-gb",
        type=float,
        required=True,
        help=(
            "Per-category size limit in GB. Each GenImage "
            "category is downloaded in full (all its split-zip "
            "volumes) only if its total size is <= this limit; "
            "categories over the limit are skipped entirely, "
            "since a partial set of volumes can't be extracted."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/genimage"),
        help=(
            "Output directory. "
            "Default: ./data/genimage"
        ),
    )

    args = parser.parse_args()

    if args.max_gb <= 0:
        parser.error(
            "--max-gb must be greater than 0"
        )

    download_genimage(
        output_dir=args.output_dir,
        max_gb=args.max_gb,
    )
