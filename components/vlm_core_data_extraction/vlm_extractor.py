import json
import time
from pathlib import Path

import torch
from pydantic import ValidationError
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from helpers import render_pdf, validate_response
from prompts import build_prompt

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
INPUT_PDF = "Joppe_Iemants_CV.pdf"

PDF_PATH = BASE_DIR / "assets" / "original_cvs" / INPUT_PDF
OUTPUT_BASE = BASE_DIR / "outputs"
OUTPUT_PATH = OUTPUT_BASE / "raw_cv.json"
RAW_RESPONSE_PATH = OUTPUT_BASE / "raw_response.txt"

PDF_DPI = 120
MAX_NEW_TOKENS = 4096

def load_model():
    print("Loading Qwen3-VL...")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )

    model.eval()
    return processor, model

def extract_cv(processor, model, images):
    print("Rendering PDF...")

    content = [ {"type": "image", "image": image} for image in images]

    content.append({
        "type": "text",
        "text": build_prompt(),
    })

    messages = [{
        "role": "user",
        "content": content,
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    print(f"Input tokens: {input_tokens}")
    print("Running structured extraction...")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = output[
        :,
        input_tokens:,
    ]

    token_count = generated_tokens.shape[1]

    print(f"Inference time: {elapsed:.2f}s")
    print(f"Generated tokens: {token_count}")
    print(f"Generation speed: {token_count / elapsed:.2f} tokens/s")
    print(
        f"Peak GPU allocation: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )

    if token_count >= MAX_NEW_TOKENS:
        print("WARNING: Generation reached the token limit!")

    return processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

def main():
    images = render_pdf(PDF_PATH, PDF_DPI)

    processor, model = load_model()

    response = extract_cv(processor, model, images)

    RAW_RESPONSE_PATH.write_text(
        response,
        encoding="utf-8",
    )

    try:
        cv = validate_response(response)
    except (json.JSONDecodeError, ValidationError) as error:
        print("Extraction failed validation:")
        print(error)
        return

    OUTPUT_PATH.write_text(
        cv.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"Validated CV saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
