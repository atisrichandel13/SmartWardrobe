"""
scripts/download_data.py

Helper to download and verify Polyvore Outfits.
DeepFashion requires a manual form submission (see instructions below).

Usage:
    python scripts/download_data.py --polyvore    # downloads Polyvore Outfits
    python scripts/download_data.py --verify      # checks all expected files exist
"""

import argparse
import os
import sys
import subprocess
import zipfile
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Expected file lists for verification
# ──────────────────────────────────────────────────────────────────────────────

DEEPFASHION_REQUIRED = [
    "Anno/list_category_img.txt",
    "Anno/list_category_cloth.txt",
    "Anno/list_attr_img.txt",
    "Anno/list_attr_cloth.txt",
    "Eval/list_eval_partition.txt",
]

POLYVORE_REQUIRED = [
    "polyvore_item_metadata.json",
    "train.json",
    "valid.json",
    "test.json",
    "fill_in_the_blank_train.json",
    "fill_in_the_blank_valid.json",
    "fill_in_the_blank_test.json",
]


def verify_deepfashion(root: str):
    root = Path(root)
    print(f"\n[DeepFashion] Checking {root}…")
    ok = True
    for rel in DEEPFASHION_REQUIRED:
        p = root / rel
        if p.exists():
            print(f"  [OK]  {rel}")
        else:
            print(f"  [MISSING]  MISSING: {rel}")
            ok = False
    img_dir = root / "img"
    if img_dir.exists():
        n = sum(1 for _ in img_dir.rglob("*.jpg"))
        print(f"  [OK]  img/ ({n:,} JPEG files found)")
    else:
        print(f"  [MISSING]  MISSING: img/")
        ok = False
    return ok


def verify_polyvore(root: str):
    root = Path(root)
    print(f"\n[Polyvore] Checking {root}…")
    ok = True
    for rel in POLYVORE_REQUIRED:
        p = root / rel
        if p.exists():
            print(f"  [OK]  {rel}")
        else:
            print(f"  [MISSING]  MISSING: {rel}")
            ok = False
    img_dir = root / "images"
    if img_dir.exists():
        n = sum(1 for _ in img_dir.glob("*.jpg"))
        print(f"  [OK]  images/ ({n:,} item images found)")
    else:
        print(f"  [MISSING]  MISSING: images/")
        ok = False
    return ok


def download_polyvore(dest: str):
    """
    Polyvore Outfits (nondisjoint split) – publicly available on Google Drive.
    We use gdown for the download.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("Installing gdown…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    # Official Google Drive file ID from Vasileva et al. (fashion-compatibility repo)
    file_id  = "1NY0fBNBEKmPcPH-6T7mHFKhLO-9HKFJ7"   # polyvore_outfits.zip
    zip_path = dest / "polyvore_outfits.zip"

    if not zip_path.exists():
        print(f"Downloading Polyvore Outfits → {zip_path} …")
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
    else:
        print(f"Archive already exists: {zip_path}")

    print("Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(dest))
    print(f"Extracted to {dest}")


def print_deepfashion_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                 DeepFashion – Manual Download Steps                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  DeepFashion requires accepting a license agreement via a form.      ║
║                                                                      ║
║  1. Go to:                                                           ║
║     http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/               ║
║     → "Category and Attribute Prediction Benchmark"                  ║
║     → Fill in the form to get the Google Drive link                  ║
║                                                                      ║
║  2. Download these archives (all are inside the Google Drive):       ║
║     • img.zip       (~30GB, all garment images)                     ║
║     • Anno.zip      (<1MB, annotation files)                        ║
║     • Eval.zip      (<1MB, train/val/test split)                    ║
║                                                                      ║
║  3. Extract all into:                                                ║
║       data/deepfashion/                                              ║
║     So the structure is:                                             ║
║       data/deepfashion/img/...                                       ║
║       data/deepfashion/Anno/list_category_img.txt                    ║
║       data/deepfashion/Eval/list_eval_partition.txt                  ║
║                                                                      ║
║  4. Run:  python scripts/download_data.py --verify                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polyvore",      action="store_true",
                        help="Download Polyvore Outfits automatically")
    parser.add_argument("--verify",        action="store_true",
                        help="Verify that all required files exist")
    parser.add_argument("--polyvore-dir",  default="data/polyvore_outfits")
    parser.add_argument("--deepfashion-dir", default="data/deepfashion")
    parser.add_argument("--deepfashion-instructions", action="store_true",
                        help="Show DeepFashion download instructions")
    args = parser.parse_args()

    if args.deepfashion_instructions:
        print_deepfashion_instructions()
        return

    if args.polyvore:
        download_polyvore(args.polyvore_dir)

    if args.verify:
        df_ok = verify_deepfashion(args.deepfashion_dir)
        pv_ok = verify_polyvore(args.polyvore_dir)
        print()
        if df_ok and pv_ok:
            print("[OK] All files verified. You are ready to train.")
        else:
            print("[MISSING] Some files are missing. See above.")
            if not df_ok:
                print("  Run: python scripts/download_data.py --deepfashion-instructions")
            sys.exit(1)

    if not any([args.polyvore, args.verify, args.deepfashion_instructions]):
        parser.print_help()


if __name__ == "__main__":
    main()
