"""
prepare_data.py — Chuyển annotation gốc của VizWiz sang định dạng train.json
mà train.py cần.

Annotation gốc VizWiz: list các dict, mỗi dict có "image", "question",
"answers" (list 10 đáp án từ 10 người gán nhãn).

Đầu ra: list các dict {"image", "question", "answer", "answers"}:
  - "answer": đáp án phổ biến nhất (majority vote) — giữ để backwards compat.
  - "answers": list các đáp án đã làm sạch — để train.py weighted-sample
    lúc runtime (khớp finetune_blip2.py của BLaVe-CoT).

Cách dùng:
  python prepare_data.py \
      --raw data/vizwiz/Annotations/train.json \
      --out data/train.json

  # Cho demo (chỉ lấy N mẫu đầu):
  python prepare_data.py --raw ... --out ... --max_samples 200
"""

import argparse
import json
import os
import sys
from collections import Counter

# Windows console mặc định cp1252 không in được tiếng Việt / ký tự ✓.
# Ép stdout sang UTF-8 ngay từ đầu để tránh UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Các đáp án "rác" cần bỏ (mô hình không nên học trả lời "không biết")
SKIP_ANSWERS = {"unanswerable", "unsuitable", "unsuitable image", ""}


def majority_answer(answers):
    """Lấy đáp án phổ biến nhất từ list các dict {'answer': ...}."""
    cleaned = [a["answer"].strip().lower() for a in answers if a.get("answer")]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def convert(raw_path, max_samples=None):
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if max_samples:
        raw = raw[:max_samples]

    out = []
    needed_images = set()
    skipped = 0

    for item in raw:
        # VizWiz dùng key "answers"; phòng trường hợp thiếu
        answers = item.get("answers", [])
        cleaned = [
            a["answer"].strip().lower()
            for a in answers
            if a.get("answer") and a["answer"].strip().lower() not in SKIP_ANSWERS
        ]

        if not cleaned:
            skipped += 1
            continue

        # Majority vote cho backwards compat + cleaned list cho weighted sampling.
        mc = Counter(cleaned).most_common(1)[0][0]

        out.append({
            "image": item["image"],
            "question": item["question"],
            "answer": mc,
            "answers": cleaned,   # ← BLaVe-CoT style: giữ cả list để train.py sample
        })
        needed_images.add(item["image"])

    return out, needed_images, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Đường dẫn Annotations/train.json gốc")
    ap.add_argument("--out", required=True, help="Đường dẫn train.json đầu ra")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Chỉ lấy N mẫu đầu (cho demo)")
    ap.add_argument("--list_images", default=None,
                    help="(Tuỳ chọn) Ghi danh sách ảnh cần dùng ra file txt")
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        raise FileNotFoundError(
            f"Không thấy file annotation: {args.raw}\n"
            f"Bạn đã tải và giải nén Annotations.zip chưa?"
        )

    out, needed_images, skipped = convert(args.raw, args.max_samples)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✓ Đã chuyển {len(out)} mẫu  (bỏ {skipped} mẫu unanswerable)")
    print(f"  Cần {len(needed_images)} ảnh duy nhất")
    print(f"  Ghi ra: {args.out}")

    # Tuỳ chọn: xuất danh sách ảnh cần thiết (hữu ích để lọc ảnh cho demo)
    if args.list_images:
        with open(args.list_images, "w") as f:
            f.write("\n".join(sorted(needed_images)))
        print(f"  Danh sách ảnh: {args.list_images}")


if __name__ == "__main__":
    main()
