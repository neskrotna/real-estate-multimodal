from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError

# HEIC/HEIF support
# install with:
# pip install pillow-heif
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


SUPPORTED_EXTENSIONS = {
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_INPUT_DIR = Path("data/raw/images")
DEFAULT_REPORT_PATH = Path("reports/image_conversion_report.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert real estate listing images to JPG and rename them consistently."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root folder containing listing folders, e.g. data/raw/images",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to save the CSV conversion report",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPG quality (1-100). Recommended: 90-95",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Optional resize: max width or height. 0 means keep original size",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JPG outputs if they already exist",
    )

    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original image files after successful conversion",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing files",
    )

    return parser.parse_args()


def ensure_report_parent(report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)


def is_supported_image(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_listing_dirs(images_root: Path) -> List[Path]:
    if not images_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {images_root}")

    listing_dirs = [p for p in images_root.iterdir() if p.is_dir()]
    listing_dirs.sort(key=lambda p: p.name)
    return listing_dirs


def collect_images_in_listing(listing_dir: Path) -> List[Path]:
    files = [p for p in listing_dir.iterdir() if is_supported_image(p)]
    files.sort(key=lambda p: p.name.lower())
    return files


def open_image_safely(image_path: Path) -> Image.Image:
    try:
        img = Image.open(image_path)
        img.load()
        return img
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Unsupported or corrupted image: {image_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to open image: {image_path}") from exc


def convert_to_rgb(img: Image.Image) -> Image.Image:
    """
    JPG does not support alpha/transparency.
    If the image has transparency, place it on a white background.
    """
    if img.mode in ("RGB",):
        return img

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.getchannel("A") if "A" in img.getbands() else None
        background.paste(img.convert("RGBA"), mask=alpha)
        return background

    if img.mode == "P":
        return img.convert("RGB")

    return img.convert("RGB")


def resize_if_needed(img: Image.Image, max_side: int) -> Tuple[Image.Image, bool]:
    if max_side <= 0:
        return img, False

    width, height = img.size
    largest_side = max(width, height)

    if largest_side <= max_side:
        return img, False

    scale = max_side / float(largest_side)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = img.resize((new_width, new_height), Image.LANCZOS)
    return resized, True


def build_output_path(listing_dir: Path, listing_id: str, index: int) -> Path:
    filename = f"{listing_id}_img_{index:02d}.jpg"
    return listing_dir / filename


def safe_delete(path: Path) -> None:
    try:
        path.unlink()
    except Exception as exc:
        print(f"[WARNING] Failed to delete original file: {path} | {exc}")


def write_report(report_path: Path, rows: List[dict]) -> None:
    ensure_report_parent(report_path)

    fieldnames = [
        "listing_id",
        "original_file",
        "new_file",
        "original_extension",
        "status",
        "resized",
        "width",
        "height",
        "message",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_environment() -> None:
    if not HEIF_AVAILABLE:
        print(
            "[INFO] pillow-heif is not installed. HEIC/HEIF files will fail.\n"
            "Install it with: pip install pillow-heif"
        )


def process_listing(
    listing_dir: Path,
    quality: int,
    max_side: int,
    overwrite: bool,
    delete_originals: bool,
    dry_run: bool,
) -> List[dict]:
    listing_id = listing_dir.name
    image_files = collect_images_in_listing(listing_dir)

    if not image_files:
        return [{
            "listing_id": listing_id,
            "original_file": "",
            "new_file": "",
            "original_extension": "",
            "status": "empty_folder",
            "resized": "",
            "width": "",
            "height": "",
            "message": "No supported images found in listing folder",
        }]

    report_rows: List[dict] = []
    new_index = 1

    for image_path in image_files:
        output_path = build_output_path(listing_dir, listing_id, new_index)

        # If the source file is already exactly the target JPG filename,
        # keep it as-is and only optionally resize/re-save if overwrite is used.
        already_target_named = (
            image_path.suffix.lower() == ".jpg"
            and image_path.name == output_path.name
        )

        if output_path.exists() and not overwrite and not already_target_named:
            report_rows.append({
                "listing_id": listing_id,
                "original_file": str(image_path),
                "new_file": str(output_path),
                "original_extension": image_path.suffix.lower(),
                "status": "skipped_exists",
                "resized": "",
                "width": "",
                "height": "",
                "message": "Output JPG already exists",
            })
            new_index += 1
            continue

        try:
            img = open_image_safely(image_path)
            img = convert_to_rgb(img)
            img, resized = resize_if_needed(img, max_side)
            width, height = img.size

            if dry_run:
                report_rows.append({
                    "listing_id": listing_id,
                    "original_file": str(image_path),
                    "new_file": str(output_path),
                    "original_extension": image_path.suffix.lower(),
                    "status": "dry_run",
                    "resized": resized,
                    "width": width,
                    "height": height,
                    "message": "No file written in dry-run mode",
                })
                new_index += 1
                continue

            img.save(output_path, format="JPEG", quality=quality, optimize=True)

            if delete_originals:
                # delete only if the original file is different from the new output
                if image_path.resolve() != output_path.resolve():
                    safe_delete(image_path)

            report_rows.append({
                "listing_id": listing_id,
                "original_file": str(image_path),
                "new_file": str(output_path),
                "original_extension": image_path.suffix.lower(),
                "status": "converted",
                "resized": resized,
                "width": width,
                "height": height,
                "message": "Success",
            })

        except Exception as exc:
            report_rows.append({
                "listing_id": listing_id,
                "original_file": str(image_path),
                "new_file": str(output_path),
                "original_extension": image_path.suffix.lower(),
                "status": "failed",
                "resized": "",
                "width": "",
                "height": "",
                "message": str(exc),
            })

        new_index += 1

    return report_rows


def print_summary(rows: List[dict]) -> None:
    total = len(rows)
    converted = sum(1 for r in rows if r["status"] == "converted")
    failed = sum(1 for r in rows if r["status"] == "failed")
    skipped_exists = sum(1 for r in rows if r["status"] == "skipped_exists")
    dry_run = sum(1 for r in rows if r["status"] == "dry_run")
    empty_folders = sum(1 for r in rows if r["status"] == "empty_folder")

    print("\n=== Conversion Summary ===")
    print(f"Total report rows:     {total}")
    print(f"Converted:             {converted}")
    print(f"Failed:                {failed}")
    print(f"Skipped existing:      {skipped_exists}")
    print(f"Dry-run entries:       {dry_run}")
    print(f"Empty folders:         {empty_folders}")


def main() -> None:
    args = parse_args()
    validate_environment()

    images_root: Path = args.input_dir
    report_path: Path = args.report_path

    if not images_root.exists():
        print(f"[ERROR] Input directory not found: {images_root}")
        sys.exit(1)

    listing_dirs = collect_listing_dirs(images_root)

    if not listing_dirs:
        print(f"[ERROR] No listing folders found inside: {images_root}")
        sys.exit(1)

    print(f"[INFO] Found {len(listing_dirs)} listing folders in {images_root}")

    all_rows: List[dict] = []

    for listing_dir in listing_dirs:
        print(f"[INFO] Processing listing: {listing_dir.name}")
        rows = process_listing(
            listing_dir=listing_dir,
            quality=args.quality,
            max_side=args.max_side,
            overwrite=args.overwrite,
            delete_originals=args.delete_originals,
            dry_run=args.dry_run,
        )
        all_rows.extend(rows)

    write_report(report_path, all_rows)
    print_summary(all_rows)
    print(f"\n[INFO] Report written to: {report_path}")

if __name__ == "__main__":
    main()