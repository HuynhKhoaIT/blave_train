"""
make_demo.py — Đóng gói một bộ demo siêu nhỏ (~vài chục MB) để upload Colab.

Tiền đề: đã chạy
    python prepare_data.py --raw data/train.json \
        --out data/train_demo.json --max_samples 200 \
        --list_images data/needed.txt

Đầu ra: `demo_pack.zip` ở thư mục hiện tại, có cấu trúc bên trong:
    data/
        train_converted.json        ← đổi tên từ train_demo.json để khớp config
        vizwiz/
            train/
                VizWiz_train_XXXXXXXX.jpg
                ...

Trên Colab chỉ cần upload file zip rồi:
    !unzip -q demo_pack.zip
    !python train.py
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

# Windows console mặc định cp1252 — ép UTF-8 để in được tiếng Việt / ✓.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
SRC_IMAGES = ROOT / "data" / "vizwiz" / "train"
SRC_JSON = ROOT / "data" / "train_demo.json"
NEEDED_TXT = ROOT / "data" / "needed.txt"

OUT_ZIP = ROOT / "demo_pack.zip"
STAGING = ROOT / "demo_pack_staging"


def main():
    # --- Kiểm tra điều kiện đầu vào ---
    for path, hint in [
        (SRC_JSON, "Chạy prepare_data.py --max_samples 200 trước."),
        (NEEDED_TXT, "Chạy prepare_data.py với --list_images data/needed.txt"),
        (SRC_IMAGES, "Cần thư mục ảnh data/vizwiz/train/"),
    ]:
        if not path.exists():
            sys.exit(f"❌ Không thấy: {path}\n   → {hint}")

    needed = [line.strip() for line in NEEDED_TXT.read_text().splitlines() if line.strip()]
    print(f"Cần đóng gói {len(needed)} ảnh.")

    # --- Dọn staging cũ, tạo cấu trúc giống Colab sẽ thấy ---
    if STAGING.exists():
        shutil.rmtree(STAGING)
    img_out_dir = STAGING / "data" / "vizwiz" / "train"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy ảnh ---
    missing = []
    copied = 0
    for name in needed:
        src = SRC_IMAGES / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, img_out_dir / name)
        copied += 1

    if missing:
        print(f"⚠ Thiếu {len(missing)} ảnh trong data/vizwiz/train/ (vd {missing[:3]}).")
        print("  Gói demo vẫn được tạo với phần ảnh có sẵn.")

    # --- Copy json với tên train_converted.json để khớp config.py mặc định ---
    json_out = STAGING / "data" / "train_converted.json"
    shutil.copy2(SRC_JSON, json_out)
    print(f"✓ Copy {copied} ảnh + train_converted.json vào {STAGING.name}/")

    # --- Nén ---
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in STAGING.rglob("*"):
            if f.is_file():
                z.write(f, arcname=f.relative_to(STAGING))

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"✓ Đã tạo {OUT_ZIP.name}  ({size_mb:.1f} MB)")
    print("\nTrên Colab chạy:")
    print("    !unzip -q demo_pack.zip")
    print("    !python train.py")

    # Dọn staging (giữ lại zip)
    shutil.rmtree(STAGING)


if __name__ == "__main__":
    main()
