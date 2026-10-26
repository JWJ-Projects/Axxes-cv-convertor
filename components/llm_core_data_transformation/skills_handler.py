"""Transform extracted CV data into renderer-ready skill data.

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
    parser.add_argument("input", type=Path, help="Validated RawCV JSON")
    parser.add_argument("output", type=Path, help="Destination skills JSON")
    parser.add_argument("--interview", type=Path, help="Optional UTF-8 transcript")
    parser.add_argument("--adapter", type=Path, help="Opt into model inference with a trained adapter")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or (
        args.interview and args.interview.resolve() == args.output.resolve()
    ):
        parser.error("Output must not overwrite an input file")
    raw_cv = RawCV.model_validate_json(args.input.read_text(encoding="utf-8"))
    transcript = args.interview.read_text(encoding="utf-8") if args.interview else None
    predictor = GLiNERProficiencyPredictor(args.adapter) if args.adapter else None
    result = transform_skills(raw_cv, transcript, predictor=predictor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {len(result.skills)} skills; "
          f"{sum(s.review_required for s in result.skills)} require review: {args.output}")


if __name__ == "__main__":
    main()
