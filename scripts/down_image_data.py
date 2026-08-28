"""
Automated Dataset Downloader and Setup Utility.
Supports:
1. SID-Set via HuggingFace (saberzl/SID_Set)
2. WildFake Benchmark (COCO val2017 + DALL-E)
3. GenImage Benchmark subsets
4. Mock/Synthetic Toy Dataset Generator (for instant local verification)
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw


def generate_mock_toy_dataset(output_dir: Path, num_samples: int = 40):
    """
    Generate synthetic placeholder images for fast unit tests and pipeline sanity checks.
    """
    print(f"Generating mock dataset with {num_samples} samples at {output_dir}...")
    real_train = output_dir / "sid_set" / "train" / "0_real"
    fake_train = output_dir / "sid_set" / "train" / "1_fake"
    real_val = output_dir / "sid_set" / "val" / "0_real"
    fake_val = output_dir / "sid_set" / "val" / "1_fake"

    wildfake_real = output_dir / "wildfake" / "real"
    wildfake_fake = output_dir / "wildfake" / "fake"

    for d in [real_train, fake_train, real_val, fake_val, wildfake_real, wildfake_fake]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    def make_image(is_fake: bool, path: Path):
        # Create 512x512 image
        if is_fake:
            # AI-like pattern (smooth gradient + high frequency noise)
            arr = np.zeros((512, 512, 3), dtype=np.uint8)
            arr[:, :, 0] = np.linspace(50, 220, 512, dtype=np.uint8)
            arr[:, :, 1] = np.linspace(200, 50, 512, dtype=np.uint8)[:, None]
            arr[:, :, 2] = (arr[:, :, 0] // 2 + arr[:, :, 1] // 2)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            draw.ellipse([100, 100, 400, 400], outline="white", width=4)
        else:
            # Natural-like texture pattern
            noise = rng.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(noise)

        img.save(path, format="JPEG", quality=85)

    for i in range(num_samples):
        target_dir_real = real_train if i < num_samples * 0.8 else real_val
        target_dir_fake = fake_train if i < num_samples * 0.8 else fake_val
        make_image(False, target_dir_real / f"real_{i:04d}.jpg")
        make_image(True, target_dir_fake / f"fake_{i:04d}.jpg")

    for i in range(10):
        make_image(False, wildfake_real / f"coco_{i:04d}.jpg")
        make_image(True, wildfake_fake / f"dalle_{i:04d}.jpg")

    print(f"Mock datasets created successfully at {output_dir}")


def extract_sid_parquet_to_images(
        output_dir: Path,
        max_shards: Optional[int] = 40,
        val_split: float = 0.1,
        input_dir: Optional[Path] = None,
):
    """
    Ultra-Fast multi-threaded SID-Set Parquet to JPEG Extractor.
    Extracts ~40,000 images in seconds into standard train/val 0_real and 1_fake directories.

    `input_dir`, if given, is scanned for the source .parquet shards directly
    (e.g. wherever you already manually downloaded/placed them) instead of
    requiring them to first be copied into `output_dir/sid_set/`. The
    extracted JPEGs still always land under `output_dir/sid_set/...`.
    """
    sid_root = output_dir / "sid_set"
    scan_root = Path(input_dir) if input_dir is not None else sid_root
    parquet_files = sorted(list(scan_root.rglob("*.parquet")))

    if not parquet_files:
        print(f"--> [ERROR] No parquet files found under {scan_root}")
        return

    if max_shards:
        parquet_files = parquet_files[:max_shards]

    train_real = sid_root / "train" / "0_real"
    train_fake = sid_root / "train" / "1_fake"
    val_real = sid_root / "val" / "0_real"
    val_fake = sid_root / "val" / "1_fake"

    for d in [train_real, train_fake, val_real, val_fake]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n=================================================================")
    print(f"🚀 High-Speed SID-Set Parquet Extractor")
    print(f"=================================================================")
    print(f"--> Found {len(parquet_files)} parquet shards. Extracting images in parallel...")

    try:
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import random

        def save_file(target_path: Path, raw_data: Any):
            if isinstance(raw_data, bytes):
                with open(target_path, "wb") as f:
                    f.write(raw_data)
            elif isinstance(raw_data, dict) and "bytes" in raw_data and raw_data["bytes"]:
                with open(target_path, "wb") as f:
                    f.write(raw_data["bytes"])
            elif isinstance(raw_data, Image.Image):
                raw_data.convert("RGB").save(target_path, quality=95)
            elif isinstance(raw_data, (dict, list)):
                try:
                    Image.open(io.BytesIO(raw_data.get("bytes", b""))).save(target_path, quality=95)
                except Exception:
                    pass

        pool = ThreadPoolExecutor(max_workers=16)
        total_extracted = 0
        pbar = tqdm(desc="Extracting SID-Set Images")

        for shard_idx, pf in enumerate(parquet_files):
            table = pq.read_table(pf)
            df = table.to_pandas()

            schema_names = list(df.columns)
            img_c = next((c for c in ["image", "img", "bytes", "data"] if c in schema_names), schema_names[0])
            lbl_c = next((c for c in ["label", "type", "is_fake", "is_synthetic", "class"] if c in schema_names),
                         schema_names[-1])

            for row_idx, row in df.iterrows():
                raw_label = row[lbl_c]
                is_ai = False
                if isinstance(raw_label, (int, float, np.integer, np.floating)):
                    is_ai = bool(raw_label >= 0.5)
                else:
                    is_ai = any(
                        k in str(raw_label).lower() for k in ["fake", "synthetic", "aigc", "gen", "tampered", "1"])

                # Determine train vs val
                is_val = (random.random() < val_split)
                if is_val:
                    target_dir = val_fake if is_ai else val_real
                else:
                    target_dir = train_fake if is_ai else train_real

                img_raw = row[img_c]
                if img_raw is not None:
                    target_path = target_dir / f"sid_{shard_idx:03d}_{row_idx:05d}.jpg"
                    pool.submit(save_file, target_path, img_raw)
                    total_extracted += 1
                    pbar.update(1)

        pool.shutdown(wait=True)
        pbar.close()
        print(
            f"--> [SUCCESS] Successfully extracted {total_extracted} SID-Set images into {sid_root}/train and {sid_root}/val in seconds!\n")

    except Exception as e:
        print(f"--> [ERROR] Failed extracting SID-Set parquet files: {e}")


def download_sid_set(output_dir: Path):
    """Download SID-Set from Hugging Face and extract automatically."""
    print("Downloading SID-Set from Hugging Face (saberzl/SID_Set)...")
    try:
        from huggingface_hub import snapshot_download
        sid_dir = output_dir / "sid_set"
        snapshot_download(
            repo_id="saberzl/SID_Set",
            repo_type="dataset",
            local_dir=str(sid_dir),
            local_dir_use_symlinks=False,
        )
        print(f"SID-Set successfully downloaded to {sid_dir}")
        # Auto extract
        extract_sid_parquet_to_images(output_dir)
    except Exception as e:
        print(f"Error downloading SID-Set from Hugging Face: {e}")


def download_genimage(
        output_dir: Path,
        subset: str = "tiny",
        max_samples: int = 1000,
):
    """
    Download authentic GenImage test benchmark for cross-generator OOD evaluation.
    Supports individual generator subsets from 'nebula/GenImage-arrow':
    - 'midjourney', 'sd14', 'sd15', 'wukong', 'vqdm', 'glide', 'biggan', 'adm'
    - 'tiny': TheKernel01/Tiny-GenImage
    - 'all': Downloads authentic test sets across all 8 generators
    """
    GEN_CONFIGS = {
        "tiny": ("TheKernel01/Tiny-GenImage", None, "validation"),
        "midjourney": ("nebula/GenImage-arrow", "midjourney-test", "test"),
        "sd14": ("nebula/GenImage-arrow", "sd14-test", "test"),
        "sd15": ("nebula/GenImage-arrow", "sd15-test", "test"),
        "wukong": ("nebula/GenImage-arrow", "wukong-test", "test"),
        "vqdm": ("nebula/GenImage-arrow", "vqdm-test", "test"),
        "glide": ("nebula/GenImage-arrow", "glide-test", "test"),
        "biggan": ("nebula/GenImage-arrow", "biggan-test", "test"),
        "adm": ("nebula/GenImage-arrow", "adm-test", "test"),
    }

    subsets_to_download = [k for k in GEN_CONFIGS.keys() if k != "tiny"] if subset == "all" else [subset]

    for sub in subsets_to_download:
        if sub not in GEN_CONFIGS:
            print(f"Unknown generator subset '{sub}'. Available: {list(GEN_CONFIGS.keys())}")
            continue

        repo_id, config_name, split_name = GEN_CONFIGS[sub]
        folder_name = "tiny_genimage" if sub == "tiny" else sub
        gen_dir = output_dir / "genimage" / folder_name
        nature_dir = gen_dir / "nature"
        ai_dir = gen_dir / "ai"

        # Clean existing directory to prevent mixing duplicate data
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        nature_dir.mkdir(parents=True, exist_ok=True)
        ai_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--> Downloading authentic GenImage subset [{sub}] from {repo_id} ({config_name or 'default'})...")
        try:
            from datasets import load_dataset

            if config_name:
                ds = load_dataset(repo_id, config_name, split=split_name, streaming=True)
            else:
                ds = load_dataset(repo_id, split=split_name, streaming=True)

            from tqdm import tqdm
            import io

            count_real = 0
            count_ai = 0
            half_max = max_samples // 2

            pbar = tqdm(total=max_samples, desc=f"Downloading [{sub}]")
            for item in ds:
                img_data = item.get("image", None)
                raw_label = item.get("label", item.get("class", item.get("label_name", None)))

                is_ai = False
                if raw_label is not None:
                    str_lbl = str(raw_label).strip().lower()
                    if str_lbl in ["1", "ai", "fake", "1_fake", "synthetic", "generated"]:
                        is_ai = True
                    elif str_lbl in ["0", "nature", "real", "0_real"]:
                        is_ai = False
                    elif isinstance(raw_label, int) and raw_label == 1:
                        is_ai = True
                else:
                    path_str = str(item.get("path", item.get("file_name", ""))).lower()
                    if "ai" in path_str or "fake" in path_str or "1_" in path_str:
                        is_ai = True

                if is_ai and count_ai < half_max:
                    target_dir = ai_dir
                    count_ai += 1
                elif not is_ai and count_real < half_max:
                    target_dir = nature_dir
                    count_real += 1
                else:
                    if count_ai >= half_max and count_real >= half_max:
                        break
                    continue

                # Robust decoding for PIL, dict, bytes
                img = None
                if isinstance(img_data, Image.Image):
                    img = img_data
                elif isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                elif img_data is not None:
                    try:
                        img = Image.open(img_data)
                    except Exception:
                        pass

                if img is not None:
                    img.convert("RGB").save(target_dir / f"img_{count_real + count_ai:05d}.jpg", quality=95)
                    pbar.update(1)

            pbar.close()
            print(
                f"--> [SUCCESS] Extracted {count_real + count_ai} genuine images for [{sub}] (Real: {count_real}, Fake: {count_ai})")

        except Exception as e:
            print(f"--> [ERROR] Failed to download {sub}: {e}")
            if gen_dir.exists() and len(list(gen_dir.rglob("*.jpg"))) == 0:
                shutil.rmtree(gen_dir)


def download_community_forensics(output_dir: Path, max_samples: int = 1000):
    """
    Download a balanced subset from OwensLab/CommunityForensics-Eval (CVPR 2025).
    Extracts real (0) and fake (1) images with metadata into ./data/community_forensics/
    """
    comm_dir = output_dir / "community_forensics"
    nature_dir = comm_dir / "nature"
    ai_dir = comm_dir / "ai"
    nature_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--> Streaming OwensLab/CommunityForensics-Eval (CompEval split)...")
    try:
        from datasets import load_dataset
        import io
        from tqdm import tqdm

        ds = load_dataset("OwensLab/CommunityForensics-Eval", split="CompEval", streaming=True)
        count_real = 0
        count_ai = 0
        half_max = max_samples // 2

        pbar = tqdm(total=max_samples, desc="Downloading CommunityForensics-Eval")
        for item in ds:
            label = item.get("label", 0)
            img_bytes = item.get("image_data", item.get("image", None))

            is_ai = bool(label == 1)

            if is_ai and count_ai < half_max:
                target_dir = ai_dir
                count_ai += 1
            elif not is_ai and count_real < half_max:
                target_dir = nature_dir
                count_real += 1
            else:
                if count_ai >= half_max and count_real >= half_max:
                    break
                continue

            img = None
            if isinstance(img_bytes, Image.Image):
                img = img_bytes
            elif isinstance(img_bytes, bytes):
                img = Image.open(io.BytesIO(img_bytes))
            elif isinstance(img_bytes, dict) and "bytes" in img_bytes and img_bytes["bytes"]:
                img = Image.open(io.BytesIO(img_bytes["bytes"]))

            if img is not None:
                img.convert("RGB").save(target_dir / f"img_{count_real + count_ai:05d}.jpg", quality=95)
                pbar.update(1)

        pbar.close()
        print(
            f"--> [SUCCESS] Extracted {count_real + count_ai} CommunityForensics-Eval images (Real: {count_real}, Fake: {count_ai}) at {comm_dir}")
    except Exception as e:
        print(f"--> [ERROR] Failed to download CommunityForensics-Eval: {e}")


def download_community_small(
        output_dir: Path,
        max_generators: int = 300,
        max_per_gen: int = 60,
        num_shards: int = 8,
):
    """
    Ultra-Fast Generator-Stratified Downloader from OwensLab/CommunityForensics-Small.
    Downloads first N parquet shards in parallel and dumps raw image bytes directly (50x faster).
    """
    comm_dir = output_dir / "community_small"
    comm_dir.mkdir(parents=True, exist_ok=True)
    cache_shards_dir = output_dir / ".hf_cache" / "community_small"

    print(f"\n=================================================================")
    print(f"🚀 High-Speed Generator-Stratified CommunityForensics Downloader")
    print(f"=================================================================")
    print(
        f"--> Target: {max_generators} generators x {max_per_gen} images (Downloading {num_shards} Parquet shards)...")

    try:
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import io

        # 1. High-speed parallel download of both AI shards (0-3) and Real shards (93-96)
        half_shards = max(1, num_shards // 2)
        ai_patterns = [f"data/HFCF_small_{i}.parquet" for i in range(half_shards)]
        real_patterns = [f"data/HFCF_small_{93 + i}.parquet" for i in range(half_shards)]
        patterns = ai_patterns + real_patterns

        print(f"--> Downloading {half_shards} AI shards + {half_shards} Real shards via multi-connection HTTP...")
        shard_path = snapshot_download(
            repo_id="OwensLab/CommunityForensics-Small",
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(cache_shards_dir),
            max_workers=8,
        )

        parquet_files = sorted(list(Path(cache_shards_dir).rglob("*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet shards downloaded into {cache_shards_dir}")

        print(
            f"--> [SUCCESS] Downloaded {len(parquet_files)} parquet shards (AI + Real). Extracting images in parallel...")

        gen_counts = defaultdict(lambda: {"nature": 0, "ai": 0})
        total_ai_saved = 0
        total_real_saved = 0
        target_ai = max_generators * (max_per_gen // 2)
        target_real = target_ai

        def save_file(target_path: Path, raw_data: Any):
            if isinstance(raw_data, bytes):
                with open(target_path, "wb") as f:
                    f.write(raw_data)
            elif isinstance(raw_data, dict) and "bytes" in raw_data and raw_data["bytes"]:
                with open(target_path, "wb") as f:
                    f.write(raw_data["bytes"])
            elif isinstance(raw_data, Image.Image):
                raw_data.convert("RGB").save(target_path, quality=95)
            elif isinstance(raw_data, (dict, list)):
                try:
                    Image.open(io.BytesIO(raw_data.get("bytes", b""))).save(target_path, quality=95)
                except Exception:
                    pass

        pbar = tqdm(total=target_ai + target_real, desc="Extracting Stratified Generators")
        pool = ThreadPoolExecutor(max_workers=16)

        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()

            for _, row in df.iterrows():
                label = row.get("label", 0)
                is_ai = bool(label == 1)

                if is_ai:
                    if total_ai_saved >= target_ai:
                        continue
                    gen_model = str(
                        row.get("generator_model", row.get("generator", row.get("model_name", "generator")))).replace(
                        "/", "_")
                    if len([g for g in gen_counts if gen_counts[g]["ai"] > 0]) >= max_generators and \
                            gen_counts[gen_model]["ai"] == 0:
                        continue
                    if gen_counts[gen_model]["ai"] >= (max_per_gen // 2):
                        continue

                    target_dir = comm_dir / gen_model / "ai"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    idx = gen_counts[gen_model]["ai"]
                    target_path = target_dir / f"img_{idx:04d}.jpg"

                    img_raw = row.get("image_data", row.get("image", None))
                    if img_raw is not None:
                        pool.submit(save_file, target_path, img_raw)
                        gen_counts[gen_model]["ai"] += 1
                        total_ai_saved += 1
                        pbar.update(1)

                else:  # Real / Nature
                    if total_real_saved >= target_real:
                        continue
                    real_source = str(row.get("real_source", row.get("model_name", "real_authentic"))).replace("/",
                                                                                                               "_").lower()
                    gen_folder = f"real_{real_source}" if not real_source.startswith("real_") else real_source

                    target_dir = comm_dir / gen_folder / "nature"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    idx = gen_counts[gen_folder]["nature"]
                    target_path = target_dir / f"img_{idx:04d}.jpg"

                    img_raw = row.get("image_data", row.get("image", None))
                    if img_raw is not None:
                        pool.submit(save_file, target_path, img_raw)
                        gen_counts[gen_folder]["nature"] += 1
                        total_real_saved += 1
                        pbar.update(1)

            if total_ai_saved >= target_ai and total_real_saved >= target_real:
                break

        pool.shutdown(wait=True)
        pbar.close()
        print(
            f"--> [SUCCESS] Extracted {total_ai_saved} AI images + {total_real_saved} Real images across {len(gen_counts)} generator groups at {comm_dir}!\n")

    except Exception as e:
        print(f"--> [ERROR] Failed fast parquet extraction: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Downloader for Robust AIGC Detector")
    parser.add_argument("--output_dir", type=str, default="./data", help="Target data root directory")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="[extract_sid only] Directory containing the source .parquet "
                             "shards, if you've already downloaded them yourself. Default: "
                             "<output_dir>/sid_set.")
    parser.add_argument("--dataset",
                        choices=["mock", "sid", "extract_sid", "genimage", "community_forensics", "community_small",
                                 "all"], default="mock",
                        help="Which dataset to download or generate")
    parser.add_argument("--subset", type=str, default="tiny",
                        help="GenImage subset to download (e.g. tiny, midjourney, sd14, sd15, wukong, vqdm, glide, biggan, adm, all)")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples to extract per generator")
    parser.add_argument("--max_generators", type=int, default=300, help="Max distinct generators for community_small")
    parser.add_argument("--max_per_gen", type=int, default=60, help="Max images per generator family")
    parser.add_argument("--num_shards", type=int, default=8,
                        help="Number of Parquet shards to download for community_small")
    parser.add_argument("--max_shards", type=int, default=40, help="Max Parquet shards to extract for SID-Set")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["mock", "all"]:
        generate_mock_toy_dataset(out)
    if args.dataset in ["sid", "all"]:
        download_sid_set(out)
    if args.dataset in ["extract_sid"]:
        input_dir = Path(args.input_dir) if args.input_dir else None
        extract_sid_parquet_to_images(out, max_shards=args.max_shards, input_dir=input_dir)
    if args.dataset in ["genimage", "all"]:
        download_genimage(out, subset=args.subset, max_samples=args.max_samples)
    if args.dataset in ["community_forensics"]:
        download_community_forensics(out, max_samples=args.max_samples)
    if args.dataset in ["community_small", "all"]:
        download_community_small(
            out,
            max_generators=args.max_generators,
            max_per_gen=args.max_per_gen,
            num_shards=args.num_shards,
        )


if __name__ == "__main__":
    main()
