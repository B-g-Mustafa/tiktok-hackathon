#!/usr/bin/env python3
"""
Download a size-limited subset of the GenImage dataset from
Google Drive.

Dependencies (install on the machine that runs this script):

    pip install gdown requests
"""

import argparse
import re
import sys
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


def download_file(file_id: str, dest_path: Path) -> bool:
    """Download one Google Drive file (by id) with resume support."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\nDownloading:\n"
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
        return False

    return result is not None


def download_genimage(
    output_dir: Path,
    max_gb: float,
):
    """
    Download up to max_gb of ZIP data from EACH GenImage
    generator category.

    Files are whole (never partially downloaded), so a category's
    total may land a little under the budget. Example: category
    has 3 GB zips and max_gb=10 -> 3 files are downloaded (9 GB);
    the 4th is skipped because it would push the total to 12 GB.
    """

    max_bytes = int(max_gb * 1024 ** 3)

    entries = get_drive_listing(GENIMAGE_URL)

    # ---------------------------------------------------------
    # Organize ZIP files by generator
    # ---------------------------------------------------------

    categories = {
        category: []
        for category in CATEGORIES
    }

    for entry in entries:

        path = entry["path"]

        # Only download ZIP files.
        if not path.lower().endswith(".zip"):
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

    # ---------------------------------------------------------
    # Show the plan
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("GenImage Download Plan")
    print("=" * 80)

    for category in sorted(categories):

        files = categories[category]

        # Sort by path so multipart ZIPs are downloaded in order.
        files.sort(
            key=lambda x: x["path"]
        )

        print(
            f"\n{category}: "
            f"{len(files)} ZIP files"
        )

    print("=" * 80)

    # ---------------------------------------------------------
    # Process categories
    # ---------------------------------------------------------

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for category in sorted(categories):

        files = categories[category]

        if not files:
            print(
                f"\n[WARNING] No ZIP files found for "
                f"{category}"
            )
            continue

        category_dir = (
            output_dir / category
        )

        print("\n")
        print("=" * 80)
        print(f"CATEGORY: {category}")
        print(f"Limit   : {max_gb:.2f} GB")
        print("=" * 80)

        downloaded_bytes = 0
        downloaded_files = 0

        for file_info in files:

            path = file_info["path"]
            file_id = file_info["id"]

            size = probe_file_size(session, file_id)

            # -------------------------------------------------
            # Decide whether this file still fits the budget.
            #
            # If we know its size, check it fits before starting.
            # If we don't, only start it when nothing has been
            # downloaded yet for this category (best effort).
            # -------------------------------------------------

            if size is not None:
                if downloaded_bytes + size > max_bytes:
                    print(
                        f"\nReached {max_gb:.2f} GB limit "
                        f"for {category}."
                    )
                    break
            elif downloaded_bytes >= max_bytes:
                print(
                    f"\nReached {max_gb:.2f} GB limit "
                    f"for {category} (size unknown for "
                    f"remaining files)."
                )
                break

            print(
                f"\n[{downloaded_files + 1}] "
                f"{Path(path).name}"
            )

            if size is not None:
                print(
                    f"Size: {human_size(size)}"
                )
            else:
                print("Size: unknown")

            dest_path = category_dir / Path(path).name

            success = download_file(
                file_id=file_id,
                dest_path=dest_path,
            )

            if not success:

                print(
                    f"[ERROR] Failed to download "
                    f"{path}"
                )

                print(
                    "You can run the script again; "
                    "gdown will resume partial files."
                )

                break

            # -------------------------------------------------
            # Count the downloaded file using its actual local
            # size (falls back to the probed size if the file
            # is somehow missing).
            # -------------------------------------------------

            if dest_path.exists():
                downloaded_bytes += dest_path.stat().st_size
            elif size is not None:
                downloaded_bytes += size

            downloaded_files += 1

            print(
                f"Category total: "
                f"{human_size(downloaded_bytes)} "
                f"/ {max_gb:.2f} GB"
            )

        print(
            f"\n[DONE] {category}"
        )

        print(
            f"Files downloaded: "
            f"{downloaded_files}"
        )

        print(
            f"Total: "
            f"{human_size(downloaded_bytes)}"
        )

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

        files = list(
            category_dir.glob("*.zip")
        )

        total = sum(
            f.stat().st_size
            for f in files
            if f.exists()
        )

        print(
            f"{category:<30} "
            f"{len(files):>3} files  "
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
            "Maximum GB to download from EACH "
            "GenImage generator category."
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
