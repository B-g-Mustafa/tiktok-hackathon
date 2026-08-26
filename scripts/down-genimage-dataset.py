from pathlib import Path
from collections import defaultdict
from PIL import Image
from datasets import load_dataset
import argparse


def download_genimage(
    output_dir: Path,
    max_samples: int = 1000,
    split: str = "validation",
    dataset_name: str = "TheKernel01/Tiny-GenImage",
):
    """
    Download max_samples images PER CATEGORY.

    Example:
        max_samples=1000

    If the dataset contains 5 categories, this downloads:
        category_1 -> 1000
        category_2 -> 1000
        category_3 -> 1000
        category_4 -> 1000
        category_5 -> 1000

    Total = 5000 images.
    """

    gen_dir = output_dir / "genimage"
    gen_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GenImage Downloader")
    print("=" * 70)
    print(f"Dataset          : {dataset_name}")
    print(f"Split            : {split}")
    print(f"Images/category  : {max_samples:,}")
    print(f"Output           : {gen_dir}")
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
    # Track number of downloaded images for each category
    # ---------------------------------------------------------

    category_counts = defaultdict(int)

    total_images = 0

    # ---------------------------------------------------------
    # Iterate through dataset
    # ---------------------------------------------------------

    for i, item in enumerate(ds):

        # -----------------------------------------------------
        # Get category
        # -----------------------------------------------------

        category = (
            item.get("category")
            or item.get("generator")
            or item.get("source")
        )

        label = item.get("label", 0)

        # -----------------------------------------------------
        # Fallback for Tiny-GenImage
        # -----------------------------------------------------

        if category is None:
            category = "ai" if label == 1 else "nature"

        category = str(category)

        # Make category safe for directory names
        category = (
            category
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        # -----------------------------------------------------
        # Skip category once it has enough images
        # -----------------------------------------------------

        if category_counts[category] >= max_samples:
            continue

        # -----------------------------------------------------
        # Get image
        # -----------------------------------------------------

        img = item.get("image")

        if img is None:
            continue

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
        # Create category directory
        # -----------------------------------------------------

        category_dir = gen_dir / category
        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Save image
        # -----------------------------------------------------

        image_number = category_counts[category]

        output_path = (
            category_dir /
            f"{category}_{image_number:06d}.jpg"
        )

        try:

            img.save(
                output_path,
                format="JPEG",
                quality=95,
            )

        except Exception as e:

            print(
                f"[WARNING] Could not save "
                f"{output_path}: {e}"
            )
            continue

        # -----------------------------------------------------
        # Update counters
        # -----------------------------------------------------

        category_counts[category] += 1
        total_images += 1

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if category_counts[category] == max_samples:

            print(
                f"[DONE] {category}: "
                f"{max_samples:,} images"
            )

        elif total_images % 100 == 0:

            print(
                f"Downloaded: {total_images:,} images | "
                f"Categories: {len(category_counts)}"
            )

        # -----------------------------------------------------
        # Check if ALL discovered categories are complete
        # -----------------------------------------------------

        if (
            category_counts
            and all(
                count >= max_samples
                for count in category_counts.values()
            )
        ):
            print("\nAll categories reached the requested limit.")
            break

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"Total images: {total_images:,}")
    print("\nImages per category:")

    for category, count in sorted(category_counts.items()):
        print(f"  {category:<30} {count:,}")

    print(f"\nOutput: {gen_dir}")
    print("=" * 70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Download N images from every GenImage category."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory. Default: ./data",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Number of images to download PER CATEGORY.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split. Default: validation",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="TheKernel01/Tiny-GenImage",
        help="Hugging Face dataset name.",
    )

    args = parser.parse_args()

    download_genimage(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split=args.split,
        dataset_name=args.dataset,
    )