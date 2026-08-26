#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


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
    """Get the complete Google Drive folder listing using gdown."""

    print("Reading Google Drive folder structure...")

    command = [
        "gdown",
        folder_url,
        "--folder",
        "--json",
        "--quiet",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ERROR: Could not list Google Drive folder.")
        print(result.stderr)
        sys.exit(1)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("ERROR: Could not parse gdown JSON output.")
        print(result.stdout[:2000])
        sys.exit(1)


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


def parse_size_from_listing(file_info):
    """
    gdown's JSON listing may contain size information depending
    on the Drive/gdown version.

    Returns bytes if available, otherwise None.
    """

    for key in ("size", "size_bytes", "file_size"):
        value = file_info.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (ValueError, TypeError):
            pass

    return None


def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def download_file(url: str, output_dir: Path):
    """Download one Google Drive file with resume support."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "gdown",
        url,
        "-O",
        str(output_dir),
        "--continue",
    ]

    print(
        f"\nDownloading:\n"
        f"  {url}\n"
        f"  -> {output_dir}"
    )

    result = subprocess.run(command)

    return result.returncode == 0


def download_genimage(
    output_dir: Path,
    max_gb: float,
):
    """
    Download up to max_gb of ZIP data from EACH GenImage
    generator category.

    Example:

        --max-gb 10

    means approximately:

        ADM                  <= 10 GB
        BigGAN               <= 10 GB
        glide                <= 10 GB
        Midjourney           <= 10 GB
        stable_diffusion_v_1_4 <= 10 GB
        stable_diffusion_v_1_5 <= 10 GB
        VQDM                 <= 10 GB
        wukong               <= 10 GB

    """

    max_bytes = int(max_gb * 1024 ** 3)

    listing = get_drive_listing(GENIMAGE_URL)

    # ---------------------------------------------------------
    # Organize ZIP files by generator
    # ---------------------------------------------------------

    categories = {
        category: []
        for category in CATEGORIES
    }

    for item in listing:

        path = item.get("path", "")
        url = item.get("url")

        if not url:
            continue

        # Only download ZIP files.
        if not path.lower().endswith(".zip"):
            continue

        category = get_category(path)

        if category is None:
            continue

        size = parse_size_from_listing(item)

        categories[category].append(
            {
                "path": path,
                "url": url,
                "size": size,
            }
        )

    # ---------------------------------------------------------
    # Download each category independently
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
            url = file_info["url"]
            size = file_info["size"]

            # -------------------------------------------------
            # If Drive did not expose size, we cannot safely
            # determine whether this file fits the limit.
            #
            # Downloading the first file is okay, but for
            # subsequent files we rely on the actual file size.
            # -------------------------------------------------

            if size is not None:

                if (
                    downloaded_bytes + size
                    > max_bytes
                ):
                    print(
                        f"\nReached {max_gb:.2f} GB limit "
                        f"for {category}."
                    )
                    break

            print(
                f"\n[{downloaded_files + 1}] "
                f"{Path(path).name}"
            )

            if size:
                print(
                    f"Size: {human_size(size)}"
                )

            success = download_file(
                url=url,
                output_dir=category_dir,
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
            # Count the downloaded file.
            #
            # If Drive did not provide size, use the actual
            # local file size.
            # -------------------------------------------------

            filename = Path(path).name

            local_file = (
                category_dir / filename
            )

            if local_file.exists():

                actual_size = (
                    local_file.stat().st_size
                )

                downloaded_bytes += actual_size

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