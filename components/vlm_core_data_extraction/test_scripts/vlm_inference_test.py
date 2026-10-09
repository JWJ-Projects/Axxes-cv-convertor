import torch
from PIL import Image
from transformers import (AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration)
import pymupdf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=torch.float16,
    attn_implementation="sdpa"
)

model.eval()

print("Model loaded!")


PDF_PATH = BASE_DIR / "assets" / "original_cvs" / "Joppe_Iemants_CV.pdf"

with pymupdf.open(PDF_PATH) as doc:
    page = doc[0]
    pix = page.get_pixmap(dpi=120)

image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "Read this CV image and identify the "
                    "candidate's name, occupation, skills "
                    "and education. Do not invent information."
                )
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print("Running inference...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

# Remove input tokens from generated output
generated_tokens = output[:, inputs["input_ids"].shape[1]:]

result = processor.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)[0]

print("\nVLM Response:")
print(result)
