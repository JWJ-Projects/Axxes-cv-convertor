"""Gather attributable evidence while keeping first-seen skill order stable."""
import re

from components.vlm_core_data_extraction.schemas import RawCV
from .schemas import SkillEvidence, StandardizedSkill
from .skill_normalizer import SkillNormalizer, key


def collect_skill_evidence(
    raw_cv: RawCV,
    interview_transcript: str | None,
    normalizer: SkillNormalizer,
) -> list[StandardizedSkill]:
    skills: dict[str, StandardizedSkill] = {}

    def add(name: str, evidence: SkillEvidence, category: str | None = None):
        name, canonical_category = normalizer.normalize(name)
        if not name:
            return
        entry = skills.setdefault(key(name), StandardizedSkill(
            name=name, category=canonical_category or category,
        ))
        if entry.category is None:
            entry.category = category
        if evidence not in entry.evidence:
            entry.evidence.append(evidence)

    for index, skill in enumerate(raw_cv.skills):
        add(skill.name, SkillEvidence(
            source="cv_skill", reference=f"skills/{index}", text=skill.name,
            source_pages=skill.source_pages,
            original_proficiency=(skill.proficiency.model_dump() if skill.proficiency else None),
        ), skill.category)

    for index, project in enumerate(raw_cv.projects):
        text = "\n".join(filter(None, [project.description, *project.tasks]))
        names = list(project.skills) + normalizer.find_names(text)
        # Also attach evidence to previously listed skills, including short names.
        names += [s.name for s in skills.values() if normalizer.mentions(text, s.name)]
        for name in names:
            add(name, SkillEvidence(
                source="cv_project", reference=f"projects/{index}",
                text=text or ", ".join(project.skills), source_pages=project.source_pages,
            ))

    # Line/sentence references are reproducible for an unchanged transcript.
    segments = re.split(r"\n+|(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])", interview_transcript or "")
    for index, segment in enumerate(segments):
        segment = segment.strip()
        if not segment:
            continue
        names = normalizer.find_names(segment)
        names += [s.name for s in skills.values() if normalizer.mentions(segment, s.name)]
        for name in names:
            add(name, SkillEvidence(
                source="interview", reference=f"segments/{index}", text=segment,
            ))
    return list(skills.values())
