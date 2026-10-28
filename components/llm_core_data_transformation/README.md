# Skills, projects and introduction transformation

This stage accepts extracted `RawCV` data and an optional plain-text interview
transcript. It returns validated `StandardizedSkills` JSON containing renderer-ready
skills and projects, with an optional introduction. It does not render PDFs or modify
the original extracted CV.

## Run from the project root

```bash
pipenv run python main_handler.py components/vlm_core_data_extraction/outputs/raw_cv.json outputs/skills.json
```

Optional interview and model inference:

```bash
pipenv run python main_handler.py components/vlm_core_data_extraction/outputs/raw_cv.json outputs/skills.json --interview interview.txt --adapter components/llm_core_data_transformation/decide_proficiency_finetuned_stage1/best
```

Alternatively use `python -m components.llm_core_data_transformation.skills_handler`.
The skills stage does not load a model unless `--adapter` is provided and
eligible textual evidence needs classification. An explicitly requested missing
adapter raises an error rather than silently using the base model.

Programmatic use:

```python
from components.llm_core_data_transformation.skills_handler import transform_skills

standardized = transform_skills(raw_cv, interview_transcript=None)
renderer_input = standardized.model_dump()
```

Project-only programmatic use is also available through `transform_projects(raw_cv)`
in `projects_handler.py`.

## Renderer contract (schema version 1.0)

```json
{
  "schema_version": "1.0",
  "projects": [
    {
      "title": "Payments platform",
      "company": "Example Client",
      "role": "Backend Developer",
      "start_date": "01/2024",
      "end_date": "Present",
      "description": "Built and maintained payment services.",
      "tasks": ["Implemented APIs"],
      "skills": ["Python", "Docker"],
      "source_pages": [2],
      "evidence": [{
        "source": "cv_project",
        "reference": "projects/0",
        "text": "{...}",
        "source_pages": [2]
      }],
      "review_required": false,
      "review_reasons": []
    }
  ],
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
- Project transformation preserves title/company, role, dates, description, tasks,
  skills and page references. It never invents missing roles, tasks or achievements.
  Missing Axxes project content is represented as empty/null and flagged for review.
- Explicit project skill names are canonicalized with the same conservative alias
  policy as the top-level skills. Technologies found in project descriptions/tasks
  may be added to that project's skills and are always flagged for verification.
- Project `evidence` is an internal audit trail. The renderer should use the direct
  project fields and omit evidence/review metadata from the published PDF.
- Evidence contains personal data. Do not log or publish it unnecessarily;
  only include renderer-required fields in the final public CV.

## Local LLM introduction generation

### One-command entrypoint test

```bash
pipenv run python main_handler.py --test-introduction
```

This explicitly opts into downloading/reusing `Qwen/Qwen2.5-3B-Instruct` in
`models/introduction` (several GB on first run). It uses the existing extracted
`components/vlm_core_data_extraction/outputs/raw_cv.json`, generates skills and an
introduction, saves `outputs/introduction_test.json`, and prints the introduction
and word count. Existing input files are not modified. Internet access is needed
for the initial download; the download helper may check Hub metadata on reruns.

Use `--test-introduction --introduction-model models/introduction` to skip the
download helper and use offline inference after weights have been downloaded.
Add `--introduction-4bit` for a supported CUDA GPU. Optional positional input/output
paths and `--interview` still work with the shortcut. Without the shortcut, the
existing CLI behavior is unchanged and input/output paths remain required.

### Manual model preparation

The optional introduction stage is in `introduction_generator.py`. A reasonable
starting model is **Qwen/Qwen2.5-3B-Instruct** (quality and speed must be evaluated
on your data). Download the weights once, explicitly, with internet access:

```bash
pipenv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='models/introduction')"
```

Then run locally; inference uses `local_files_only=True`, never a hosted API:

```bash
pipenv run python main_handler.py components/vlm_core_data_extraction/outputs/raw_cv.json outputs/cv_data.json --introduction-model models/introduction
```

Add `--interview interview.txt` for transcript evidence. Add
`--introduction-4bit` on a supported CUDA GPU to reduce memory requirements.
Without it, the backend uses FP16 when CUDA is available, FP32 otherwise, and
Accelerate device mapping. CPU generation works but can be slow; full FP32
3B weights alone need roughly 12 GB of RAM, plus runtime overhead. Model loading
can fail if memory is insufficient. The GLiNER model is released before loading
the introduction model when both stages are requested.

The output retains `schema_version` and `skills`, and adds `introduction`:

```json
{
  "text": "I am a software developer.",
  "sentences": [
    {"text": "I am a software developer.", "evidence_ids": ["cv:occupation"]}
  ],
  "evidence": [
    {"id": "cv:occupation", "text": "Software developer", "source_pages": []}
  ],
  "model": "models/introduction",
  "review_required": true,
  "review_reasons": ["LLM-generated introduction: verify factual support and approve before publication"]
}
```

This is the `introduction` field, not the whole output. The renderer uses
`introduction.text`; internal sentences/evidence are for review, not publication.
The combined contract is `StandardizedCV`; skills-only commands still return
`StandardizedSkills` with no extra field.

Programmatic use (inject a fake generator in tests):

```python
from components.llm_core_data_transformation.introduction_generator import (
    LocalIntroductionLLM, StandardizedCV, generate_introduction,
)

intro = generate_introduction(
    raw_cv,
    generator=LocalIntroductionLLM("models/introduction"),
    model_name="models/introduction",
    interview_transcript=None,
)
combined = StandardizedCV(**standardized.model_dump(), introduction=intro)
```

### Grounding and failure policies

- Uses original role/introduction, skill names, projects, education, certificates,
  training and optional interview lines, not inferred proficiency ratings.
- Targets English, first-person prose of 70-120 words (aiming for about 100),
  with at most six sentence entries. Sparse evidence may yield less. A hard
  140-word limit prevents oversized introductions. The prompt additionally requires
  exactly one sentence per entry; semantic sentence segmentation is not validated.
- The prompt incorporates the About Me guidance from slides 5 and 9-15 of
  `PPT_How To Make An Awesome CV (1).pdf`: a coherent elevator pitch, authentic
  professional voice, supported strengths and ambitions, positive tone and no
  cliches. These refinements supplement the original grounding/output instructions;
  example CVs and competency lists never establish facts about a candidate.
- Every generated sentence must cite existing evidence IDs. Unknown IDs, empty
  citations, invalid JSON, markup, empty evidence, oversized input and generation
  hitting the token limit fail before the output file is written. No silent
  truncation or fabricated fallback. Small local models may require prompt/model
  tuning to produce the required JSON consistently.
- **Citation validity is not semantic verification.** A real ID does not prove a
  sentence is entailed by its evidence. All generated introductions require
  human review. Interview claims are self-reported, not independently verified;
  interviewer questions, contradictions and prompt injection require attention.
- Structured name/contact/birth/location fields are excluded. Free-text evidence
  can still contain personal data; it is not automatically anonymized. Do not
  publish the audit evidence or commit private inputs to source control.
- The context budget is capped at 8,192 tokens, including 1,024 generation tokens
  (or the model's lower configured context limit). Long transcripts must be
  shortened/curated first; evidence retrieval can be added later.
- Generation is greedy (`do_sample=False`); this reduces variation but does not
  guarantee bit-for-bit reproducibility across hardware/library versions. No
  fine-tuning is performed by this stage.

## Tests

```bash
pipenv run python -m unittest discover -s tests -v
```

Tests run offline with fake generators, without loading LLM weights or changing
datasets. Real model quality, GPU compatibility and factual grounding still need
an end-to-end evaluation on your deployment hardware.
