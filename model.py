"""
model.py — Dựng mô hình BLIP-2 và gắn adapter LoRA.

Tách riêng phần này để train.py (huấn luyện) và infer.py (dùng lại) chia sẻ
cùng một logic dựng mô hình, tránh lệch nhau.

Triết lý (theo bài báo BLaVe-CoT):
  - Đóng băng Vision Encoder và Language Model (OPT-2.7B)
  - Chỉ huấn luyện adapter LoRA gắn vào Query/Key của Q-Former
  => số tham số train < 1%, file adapter chỉ ~20MB
"""
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def build_processor(base_model):
    return Blip2Processor.from_pretrained(base_model)


def build_model_for_training(cfg):
    """Dựng BLIP-2 + LoRA để huấn luyện. Trả về (model, processor)."""
    base_model = cfg["base_model"]
    processor = build_processor(base_model)

    if cfg["use_qlora"]:
        # Nạp ở 4-bit cho GPU yếu (<=12-16GB)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", #NormalFloat4 : Loại quantization tốt cho LLM.
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, # Tiết kiệm VRAM
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            base_model,
            quantization_config=bnb,
            torch_dtype=torch.float16,
            device_map="auto",   # bnb yêu cầu; cũng để train.py KHÔNG cần .to(device)
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # GPU mạnh: nạp float16 bình thường
        model = Blip2ForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.float16
        )

#Cấu hình adapter
    lora_config = LoraConfig(
        r=cfg["lora"]["r"], #rank càng lớn học càng mạnh -> tổn ram :)
        lora_alpha=cfg["lora"]["lora_alpha"], #Scale độ ảnh hưởng của LoRA. -> thường r or 2r
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"], # quan trọng để biến gắn LoRA vào layer nào
        bias="none",
    )

    try:
        model = get_peft_model(model, lora_config) #chèn lora vào model
    except ValueError as e:
        # Lỗi phổ biến nhất: tên lớp Q/K khác giữa các phiên bản transformers
        raise RuntimeError(
            "Không gắn được LoRA — nhiều khả năng target_modules sai tên.\n"
            "Hãy chạy:  python inspect_model.py\n"
            "để xem tên lớp Query/Key thật, rồi cập nhật LORA['target_modules'] trong config.py.\n"
            f"Lỗi gốc: {e}"
        )

    model.print_trainable_parameters()
    return model, processor


def load_finetuned_model(base_model, adapter_dir):
    """Nạp lại mô hình đã train (để inference): BLIP-2 gốc + adapter LoRA."""
    from peft import PeftModel
    processor = build_processor(base_model)
    base = Blip2ForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor
