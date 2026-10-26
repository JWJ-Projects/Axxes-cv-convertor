"""Transform CV data into skills and an optional evidence-grounded introduction.

Run from the project root:
    python -m components.llm_core_data_transformation.skills_handler raw_cv.json skills.json
"""
import argparse
from pathlib import Path

from components.vlm_core_data_extraction.schemas import RawCV
from .proficiency_resolver import GLiNERProficiencyPredictor, ProficiencyPredictor, predict
from .schemas import StandardizedSkills
from .skill_evidence import collect_skill_evidence
from .skill_normalizer import SkillNormalizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def prepare_test_model() -> str:
    """Explicit test-only download; Hugging Face reuses previously downloaded files."""
    from huggingface_hub import snapshot_download

    destination = PROJECT_ROOT / "models" / "introduction"
    print(f"Preparing {TEST_MODEL_ID} in {destination} (first download is several GB)...")
    snapshot_download(repo_id=TEST_MODEL_ID, local_dir=str(destination))
    return str(destination)


def transform_skills(
    raw_cv: RawCV,
    interview_transcript: str | None = None,
    *,
    predictor: ProficiencyPredictor | None = None,
    taxonomy_path: Path | None = None,
) -> StandardizedSkills:
    normalizer = SkillNormalizer() if taxonomy_path is None else SkillNormalizer(taxonomy_path)
    collected = collect_skill_evidence(raw_cv, interview_transcript, normalizer)
    return StandardizedSkills(skills=[predict(skill, predictor) for skill in collected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Validated RawCV JSON")
    parser.add_argument("output", nargs="?", type=Path, help="Destination renderer-ready JSON")
    parser.add_argument(
        "--test-introduction", action="store_true",
        help="Test with the saved CV; downloads/reuses Qwen weights unless --introduction-model is supplied",
    )
    parser.add_argument("--interview", type=Path, help="Optional UTF-8 transcript")
    parser.add_argument("--adapter", type=Path, help="Opt into model inference with a trained adapter")
    parser.add_argument("--introduction-model", help="Local/cached chat model for introduction generation")
    parser.add_argument("--introduction-4bit", action="store_true", help="Quantize the introduction model (CUDA)")
    args = parser.parse_args()
    if args.test_introduction:
        args.input = args.input or PROJECT_ROOT / "components/vlm_core_data_extraction/outputs/raw_cv.json"
        args.output = args.output or PROJECT_ROOT / "outputs/introduction_test.json"
    elif args.input is None or args.output is None:
        parser.error("input and output are required unless --test-introduction is used")
    if args.introduction_4bit and not (args.introduction_model or args.test_introduction):
        parser.error("--introduction-4bit requires --introduction-model")
    if args.input.resolve() == args.output.resolve() or (
        args.interview and args.interview.resolve() == args.output.resolve()
    ):
        parser.error("Output must not overwrite an input file")
    raw_cv = RawCV.model_validate_json(args.input.read_text(encoding="utf-8"))
    transcript = args.interview.read_text(encoding="utf-8") if args.interview else None
    if args.test_introduction and not args.introduction_model:
        # Validate inputs before spending time/bandwidth downloading weights.
        from .introduction_generator import collect_introduction_evidence
        if not collect_introduction_evidence(raw_cv, transcript):
            parser.error("No professional evidence available for the introduction test")
        args.introduction_model = prepare_test_model()
    predictor = GLiNERProficiencyPredictor(args.adapter) if args.adapter else None
    result = transform_skills(raw_cv, transcript, predictor=predictor)
    if args.introduction_model:
        from .introduction_generator import LocalIntroductionLLM, StandardizedCV, generate_introduction

        # Separate model stages to avoid retaining both GPU models unnecessarily.
        if predictor is not None and predictor.model is not None:
            predictor.model = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        introduction = generate_introduction(
            raw_cv, interview_transcript=transcript,
            generator=LocalIntroductionLLM(args.introduction_model, load_in_4bit=args.introduction_4bit),
            model_name=args.introduction_model,
        )
        result = StandardizedCV(**result.model_dump(), introduction=introduction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {len(result.skills)} skills; "
          f"{sum(s.review_required for s in result.skills)} require review: {args.output}")
    if args.introduction_model:
        print("Introduction included; factual review required before publication.")
        if args.test_introduction:
            print(f"\n--- Generated introduction ({len(result.introduction.text.split())} words) ---")
            print(result.introduction.text)
            print("Review the sentence-level evidence references in the output JSON.")


if __name__ == "__main__":
    main()
