"""
inspect_model.py — Công cụ gỡ lỗi tên lớp LoRA.

CHẠY FILE NÀY KHI: train.py báo lỗi "target modules not found" hoặc
RuntimeError nhắc bạn kiểm tra target_modules.

Lý do: tên các lớp Query/Key bên trong Q-Former khác nhau giữa các phiên bản
thư viện transformers. File này in ra tên thật để bạn điền đúng vào
LORA['target_modules'] trong config.py.

Cách dùng:
  python inspect_model.py
"""

import torch
from transformers import Blip2ForConditionalGeneration
from config import BASE_MODEL


def main():
    print(f"Đang nạp {BASE_MODEL} để kiểm tra tên lớp... (chỉ cần CPU)")
    model = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    )

    print("\nCác lớp Linear trong Q-Former có chứa 'quer' hoặc 'key' trong tên:")
    print("-" * 60)
    found = set()
    for name, module in model.named_modules():
        lname = name.lower()
        if "qformer" in lname and isinstance(module, torch.nn.Linear):
            if "quer" in lname or "key" in lname or "value" in lname:
                # Lấy phần tên cuối cùng (tên module dùng cho target_modules)
                leaf = name.split(".")[-1]
                found.add(leaf)
                print(f"  {name}   →  leaf name: '{leaf}'")

    print("-" * 60)
    if found:
        print(f"\nGỢI Ý: thử đặt trong config.py:")
        print(f"  'target_modules': {sorted(found)}")
        print("\nThường chỉ cần ['query', 'key'] hoặc tên leaf tương ứng ở trên.")
    else:
        print("\n⚠ Không tìm thấy lớp nào khớp. Hãy in toàn bộ tên module:")
        print("  for n, _ in model.named_modules(): print(n)")
        print("rồi tìm các lớp attention trong qformer thủ công.")


if __name__ == "__main__":
    main()
