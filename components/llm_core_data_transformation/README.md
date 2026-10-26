# Skills transformation

This stage accepts extracted `RawCV` data and an optional plain-text interview
transcript. It returns validated `StandardizedSkills` JSON for the CV renderer.
It does not render PDFs or modify the original extracted CV.

## Run from the project root

```bash
pipenv run python main_handler.py components/vlm_core_data_extraction/outputs/raw_cv.json outputs/skills.json
```

Optional interview and model inference:

```bash
pipenv run python main_handler.py components/vlm_core_data_extraction/outputs/raw_cv.json outputs/skills.json --interview interview.txt --adapter components/llm_core_data_transformation/decide_proficiency_finetuned_stage1/best
```

Alternatively use `python -m components.llm_core_data_transformation.skills_handler`.
No model is imported/downloaded unless `--adapter` is provided and eligible
textual evidence needs classification. An explicitly requested missing adapter
raises an error rather than silently using the base model.

Programmatic use:

```python
from components.llm_core_data_transformation.skills_handler import transform_skills

standardized = transform_skills(raw_cv, interview_transcript=None)
renderer_input = standardized.model_dump()
```

## Renderer contract (schema version 1.0)

```json
{
  "schema_version": "1.0",
  "skills": [
    {
      "name": "Python",
      "category": "Programming Languages",
      "proficiency": {
        "label": "Advanced",
        "level": 3,
        "maximum": 4,
        "method": "explicit_label"
      },
      "evidence": [
        {
          "source": "cv_skill",
          "reference": "skills/0",
          "text": "Python",
          "source_pages": [1],
          "original_proficiency": {
            "type": "text",
            "raw_value": "Advanced",
            "value": null,
            "maximum": null
          }
        }
      ],
      "review_required": false,
      "review_reasons": []
    }
  ]
}
```

The fixed mapping is Junior=1, Confirmed=2, Advanced=3, Expert=4.
A renderer draws `level` filled bars out of four. Unknown proficiency is `null`,
NOT zero or Junior. Items requiring review should be resolved before publishing;
if previewed, display them as pending rather than silently inventing a rating.
The renderer owns column splitting, label wrapping, fonts, colors and pagination.
List order follows first appearance (CV skills, projects, then interview).

## Policies and limitations for review

- Normalize canonical taxonomy names and a small reviewed alias set: TS,
  JS, Golang. Merge duplicates case-insensitively. Preserve unknown skills.
- Do NOT treat all taxonomy consolidated terms as aliases: many are related
  products rather than synonyms (e.g. Spring MVC and Spring Boot).
- Discover canonical technology names conservatively in project descriptions,
  tasks and transcript segments. Short names (Go, R, TS, etc.) must first be
  explicitly listed in CV skills/project skills; discovery skips them to reduce
  false positives. This rule-based extraction is a baseline, not semantic NER.
- Explicit exact proficiency labels and integer ratings on a four-point scale
  map directly. Five-star ratings, percentages and other scales require an
  approved conversion policy; they remain unknown for now.
- Conflicting explicit ratings remain unknown and are flagged, not averaged.
- Explicit ratings take precedence over model inference. Free-text contradictory
  claims are not automatically detected; this needs semantic conflict detection
  or human review.
- Optional GLiNER inference receives the target skill and its associated project
  or interview text, never the whole candidate profile. Short/bare evidence is
  excluded using a five-word minimum heuristic. Longer technology lists and
  negated/hypothetical mentions may still pass: every model prediction requires
  review. No uncalibrated confidence scores are exposed.
- The training examples currently contain descriptions, not the target/evidence
  wrapper used at inference. Evaluate this formatting on held-out examples before
  deployment; retrain with skill-specific context if necessary.
- Evidence contains personal data. Do not log or publish it unnecessarily;
  only include renderer-required fields in the final public CV.

## Tests

```bash
pipenv run python -m unittest discover -s tests -v
```

Tests run offline without loading GLiNER or changing datasets.
