import json
import time
from pathlib import Path

import torch
from pydantic import ValidationError
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from schemas import RawCV
from helpers import render_pdf, validate_response

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
INPUT_PDF = "Joppe_Iemants_CV.pdf"

PDF_PATH = BASE_DIR / "assets/original_cvs" / INPUT_PDF
OUTPUT_PATH = BASE_DIR / "raw_cv.json"
RAW_RESPONSE_PATH = BASE_DIR / "raw_response.txt"

PDF_DPI = 120
MAX_NEW_TOKENS = 4096

def load_model():
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


def build_prompt() -> str:
    schema = RawCV.model_json_schema()

    EXTRACTION_RULES = """
    Additional extraction rules:

    - Extract project titles separately from company names.
    - Never interpret a date as a company name.
    - Preserve education descriptions and coursework.
    - Extract phone numbers and email addresses as contact details,
      not as weblinks.
    - Preserve original skill category headings when present.
    - If a skill entry is a spoken language, put it in languages
      rather than technical skills.
    - If a language has only one general proficiency rating,
      store it in overall_level. Do not assume it applies
      exclusively to speaking, writing or understanding.
    - Preserve original text without adding professional claims.
    - Do not invent missing information to complete the schema.
    - Identify the boundaries between separate CV entries.
    - Distinguish project titles, job titles, organizations and dates.
    - If a project has no explicitly named company, set company to null.
    - Do not infer skills or tasks from a project description.
    - Preserve exact names and titles as written in the document.
    - Do not summarize or paraphrase descriptions.
    """

    return (
        "You are an information extraction system.\n"
        "Extract all available candidate information from "
        "the provided CV pages.\n\n"
        "Rules:\n"
        "1. Extract only information present in the CV.\n"
        "2. Do not invent or infer missing details.\n"
        "3. Preserve original descriptions and tasks.\n"
        "4. Use null for missing optional values.\n"
        "5. Use empty arrays for missing sections.\n"
        "6. Keep languages separate from technical skills.\n"
        "7. Associate skills and tasks with their correct projects.\n"
        "8. Preserve original proficiency indicators.\n"
        "9. Page numbers are 1-based.\n"
        "10. Return only a valid JSON object.\n\n"

        f"{EXTRACTION_RULES}\n\n"

        "The JSON must conform to this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )




def extract_cv(processor, model, images):
    content = [
        {"type": "image", "image": image}
        for image in images
    ]

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
    print("Rendering PDF...")
    images = render_pdf(PDF_PATH, PDF_DPI)
    print(f"Rendered {len(images)} page(s)")

    print("Loading Qwen3-VL...")
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
