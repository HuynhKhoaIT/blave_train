"""
train.py — Vòng huấn luyện chính cho LoRA BLIP-2 trên VizWiz.

KHÔNG cần sửa file này. Mọi thông số nằm trong config.py.

Cách dùng:
  1. Sửa PROFILE trong config.py ("demo" hoặc "full")
  2. Chạy:  python train.py

Kết quả: adapter LoRA (~20MB) được lưu vào output_dir trong config.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import get_active_config
from model import build_model_for_training


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class VizWizDataset(Dataset):
    def __init__(self, json_path, image_dir, processor, cfg, max_samples=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_samples:
            self.data = self.data[:max_samples]
        self.image_dir = image_dir
        self.processor = processor
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        it = self.data[idx]
        img_path = os.path.join(self.image_dir, it["image"])
        image = Image.open(img_path).convert("RGB")

        prompt = f"Question: {it['question']} Answer:"
        answer = it["answer"]
        full_text = f"{prompt} {answer}"

        tokenizer = self.processor.tokenizer
        max_len = self.cfg["max_prompt_len"] + self.cfg["max_answer_len"]

        # Ảnh: dùng image_processor riêng để khỏi vướng tham số padding của tokenizer.
        pixel_values = self.processor.image_processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Text: prompt + answer trong cùng chuỗi (causal LM cần đủ context để tính loss).
        full_enc = tokenizer(
            full_text, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_len,
        )
        input_ids = full_enc.input_ids.squeeze(0)
        attention_mask = full_enc.attention_mask.squeeze(0)

        # Đo độ dài prompt (không pad) để biết tới đâu là vùng cần mask khi tính loss.
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", padding=False,
            truncation=True, max_length=self.cfg["max_prompt_len"],
        ).input_ids.squeeze(0)
        prompt_len = prompt_ids.shape[0]

        # labels = input_ids, nhưng mask phần prompt và phần pad bằng -100
        # → loss chỉ tính trên các token thuộc đáp án.
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[input_ids == tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------
def main():
    cfg = get_active_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"PROFILE          : {cfg['profile_name']}")
    print(f"Device           : {device}")
    print(f"QLoRA (4-bit)    : {cfg['use_qlora']}")
    print(f"Batch size       : {cfg['batch_size']}  x grad_accum {cfg['grad_accum']}"
          f"  = effective {cfg['batch_size'] * cfg['grad_accum']}")
    print(f"Epochs           : {cfg['num_epochs']}")
    print(f"Max train samples: {cfg['max_train_samples']}")
    print("=" * 60)

    if device == "cpu":
        print("⚠ CẢNH BÁO: Không tìm thấy GPU. Train trên CPU sẽ cực kỳ chậm.")

    # --- Mô hình + LoRA ---
    model, processor = build_model_for_training(cfg)
    # Model 4-bit (qlora) đã được đặt device qua device_map="auto" lúc nạp;
    # gọi .to() trên model 4-bit sẽ raise ValueError.
    if not cfg["use_qlora"]:
        model.to(device)
    model.train()

    # --- Dữ liệu ---
    train_path = cfg["paths"]["train_json"]
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Không thấy {train_path}. Chạy prepare_data.py trước để tạo file này."
        )

    dataset = VizWizDataset(
        train_path, cfg["paths"]["image_dir"], processor, cfg,
        max_samples=cfg["max_train_samples"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    print(f"Số mẫu train: {len(dataset)}  |  Số batch/epoch: {len(loader)}")

    # --- Optimizer (chỉ tham số LoRA có requires_grad=True) ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
    )

    accum = cfg["grad_accum"]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # --- Vòng huấn luyện ---
    for epoch in range(cfg["num_epochs"]):
        running = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            running += loss.item() * accum

            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg = running / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{cfg['num_epochs']}  loss = {avg:.4f}")

        # Lưu checkpoint định kỳ
        if (epoch + 1) % cfg["save_every"] == 0:
            ckpt = os.path.join(cfg["output_dir"], f"epoch_{epoch + 1}")
            model.save_pretrained(ckpt)
            print(f"  ↳ đã lưu checkpoint: {ckpt}")

    # --- Lưu bản cuối ---
    final = os.path.join(cfg["output_dir"], "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ Hoàn tất. Adapter cuối lưu tại: {final}")
    print("  (file adapter_model.safetensors chỉ vài chục MB — đúng như bản gốc)")


if __name__ == "__main__":
    main()
