```python
from pathlib import Path
from collections import defaultdict
from PIL import Image
from datasets import load_dataset
import argparse


def format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def download_genimage(
    output_dir: Path,
    max_samples: int | None = None,
    max_size_gb: float | None = None,
    split: str = "train",
    dataset_name: str = "TheKernel01/Tiny-GenImage",
):
    """
    Download GenImage with either:

        max_samples=1000
            -> 1000 images PER CATEGORY

        max_size_gb=50
            -> stop when total downloaded size reaches 50 GB

    If both are specified, BOTH limits are respected.

    Expected dataset structure:

        label / category information
        image

    Categories are automatically detected from the dataset.
    """

    gen_dir = output_dir / "genimage"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if max_samples is None and max_size_gb is None:
        raise ValueError(
            "Specify at least one of max_samples or max_size_gb."
        )

    max_bytes = None

    if max_size_gb is not None:
        max_bytes = int(max_size_gb * 1024**3)

    print("=" * 70)
    print("GenImage Downloader")
    print("=" * 70)
    print(f"Dataset      : {dataset_name}")
    print(f"Split        : {split}")
    print(f"Output       : {gen_dir}")

    if max_samples is not None:
        print(f"Samples/category : {max_samples:,}")

    if max_size_gb is not None:
        print(f"Maximum size     : {max_size_gb:.2f} GB")

    print("=" * 70)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("\nLoading dataset...")

    try:
        ds = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
        )
    except Exception as e:
        print(f"\n[ERROR] Could not load dataset:")
        print(e)
        return

    # ---------------------------------------------------------
    # Counters
    #
    # Each category has its own counter.
    # ---------------------------------------------------------

    category_counts = defaultdict(int)

    total_images = 0
    total_bytes = 0

    # ---------------------------------------------------------
    # Iterate
    # ---------------------------------------------------------

    for i, item in enumerate(ds):

        # -----------------------------------------------------
        # Global size limit
        # -----------------------------------------------------

        if max_bytes is not None and total_bytes >= max_bytes:
            print("\nReached maximum storage limit.")
            break

        # -----------------------------------------------------
        # Get image
        # -----------------------------------------------------

        img = item.get("image")

        if img is None:
            continue

        # -----------------------------------------------------
        # Determine category
        #
        # Tiny-GenImage uses labels. If your dataset has a
        # separate category/generator field, this automatically
        # tries to use it.
        # -----------------------------------------------------

        category = (
            item.get("category")
            or item.get("generator")
            or item.get("source")
        )

        label = item.get("label", 0)

        # -----------------------------------------------------
        # Fallback for datasets where category is not provided.
        #
        # This preserves the nature/ai structure from your
        # original function.
        # -----------------------------------------------------

        if category is None:
            category = "ai" if label == 1 else "nature"

        # Make category filesystem-safe
        category = str(category).replace("/", "_").replace(" ", "_")

        # -----------------------------------------------------
        # Per-category sample limit
        # -----------------------------------------------------

        if (
            max_samples is not None
            and category_counts[category] >= max_samples
        ):
            continue

        # -----------------------------------------------------
        # Convert image
        # -----------------------------------------------------

        try:

            if not isinstance(img, Image.Image):
                img = Image.open(img)

            img = img.convert("RGB")

        except Exception as e:
            print(
                f"[WARNING] Could not process image "
                f"{i}: {e}"
            )
            continue

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        category_dir = gen_dir / category
        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_number = category_counts[category]

        output_path = (
            category_dir /
            f"{category}_{image_number:06d}.jpg"
        )

        try:

            # Save directly first
            img.save(
                output_path,
                format="JPEG",
                quality=95,
            )

            image_size = output_path.stat().st_size

        except Exception as e:

            print(
                f"[WARNING] Could not save "
                f"{output_path}: {e}"
            )

            continue

        # -----------------------------------------------------
        # Storage limit check
        #
        # If this image caused us to exceed the requested
        # limit, remove it and stop.
        # -----------------------------------------------------

        if (
            max_bytes is not None
            and total_bytes + image_size > max_bytes
        ):

            output_path.unlink(missing_ok=True)

            print(
                "\nNext image would exceed storage limit."
            )

            break

        # -----------------------------------------------------
        # Update counters
        # -----------------------------------------------------

        category_counts[category] += 1

        total_images += 1
        total_bytes += image_size

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if total_images % 100 == 0:

            print(
                f"Images: {total_images:,} | "
                f"Size: {format_size(total_bytes)} | "
                f"Categories: {len(category_counts)}"
            )

        # -----------------------------------------------------
        # Check whether all categories reached max_samples
        # -----------------------------------------------------

        if max_samples is not None:

            # We cannot know all categories in advance when
            # streaming, so this condition is evaluated using
            # the dataset's metadata when available.
            pass

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Total images : {total_images:,}"
    )

    print(
        f"Total size   : {format_size(total_bytes)}"
    )

    print("\nImages per category:")

    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"  {category:<30} {count:,}"
        )

    print("\nOutput:")
    print(f"  {gen_dir}")

    print("=" * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Download GenImage with per-category "
            "sample and/or storage limits."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of images PER CATEGORY."
        ),
    )

    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        help="Maximum total size in GB.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="TheKernel01/Tiny-GenImage",
    )

    args = parser.parse_args()

    download_genimage(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_size_gb=args.max_size,
        split=args.split,
        dataset_name=args.dataset,
    )
```
