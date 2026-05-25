"""
infer.py — Chạy thử mô hình đã train trên vài ảnh để kiểm tra chất lượng.

Mục đích: SAU KHI train, đừng tin ngay. Chạy file này để xem mô hình sinh
đáp án thế nào trên dữ liệu thật, và (tuỳ chọn) so với BLIP-2 gốc chưa train.

Cách dùng:
  python infer.py --adapter ./blip2_lora_demo/final \
                  --image data/vizwiz/val/some.jpg \
                  --question "What is this?"

  # So sánh trước/sau fine-tune:
  python infer.py --adapter ... --image ... --question ... --compare
"""

import argparse
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel
from config import BASE_MODEL


def generate(model, processor, image_path, question, device):
    image = Image.open(image_path).convert("RGB")
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=20, num_beams=3)
    # BLIP-2 + OPT trả về cả prompt prefix trong output → slice bỏ để chỉ giữ phần sinh thêm.
    generated = ids[:, prompt_len:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Thư mục adapter LoRA đã train")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--compare", action="store_true",
                    help="So sánh với BLIP-2 gốc chưa fine-tune")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(BASE_MODEL)

    base = Blip2ForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to(device)

    if args.compare:
        ans_base = generate(base, processor, args.image, args.question, device)
        print(f"[BLIP-2 GỐC]      {ans_base}")

    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()
    ans_ft = generate(model, processor, args.image, args.question, device)
    print(f"[SAU FINE-TUNE]   {ans_ft}")


if __name__ == "__main__":
    main()
