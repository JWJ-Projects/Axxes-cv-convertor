"""Transform extracted project evidence into the renderer-ready projects contract."""
import json
import re
from pathlib import Path

from components.vlm_core_data_extraction.schemas import Project, RawCV
from .schemas import ProjectEvidence, StandardizedProject, StandardizedProjects
from .skill_normalizer import SkillNormalizer, key


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip() # \s = whitespace
    return cleaned or None


def _unique(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            result.append(cleaned)
    return result


def _project_skills(
    project: Project,
    raw_cv: RawCV,
    normalizer: SkillNormalizer,
) -> tuple[list[str], list[str]]:
    """Return canonical skills and the subset conservatively found in prose."""
    explicit = []
    seen = set()
    for value in project.skills:
        name, _ = normalizer.normalize(value)
        if name and key(name) not in seen:
            seen.add(key(name))
            explicit.append(name)

    text = "\n".join(filter(None, [project.description, *project.tasks]))
    discovered = normalizer.find_names(text)
    # Allow short/ambiguous names only when the CV explicitly lists that skill.
    discovered += [skill.name for skill in raw_cv.skills if normalizer.mentions(text, skill.name)]

    inferred = []
    for value in discovered:
        name, _ = normalizer.normalize(value)
        if name and key(name) not in seen:
            seen.add(key(name))
            explicit.append(name)
            inferred.append(name)
    return explicit, inferred


def transform_projects(
    raw_cv: RawCV,
    *,
    taxonomy_path: Path | None = None,
) -> StandardizedProjects:
    """Preserve project facts, normalize lists, and flag missing renderer content.

    No role, task, date, or achievement is inferred. Technologies mentioned in a
    project's own description/tasks are conservatively added to its skills list and
    explicitly flagged for review.
    """
    normalizer = SkillNormalizer() if taxonomy_path is None else SkillNormalizer(taxonomy_path)
    projects = []
    for index, project in enumerate(raw_cv.projects):
        skills, inferred_skills = _project_skills(project, raw_cv, normalizer)
        reasons = []
        if not (_clean(project.title) or _clean(project.company)):
            reasons.append("Missing project title/company heading")
        if not _clean(project.role):
            reasons.append("Missing project role")
        if not _clean(project.description):
            reasons.append("Missing project description")
        if not project.tasks:
            reasons.append("Missing project tasks")
        if not skills:
            reasons.append("Missing project skills")
        if inferred_skills:
            reasons.append(
                "Skills discovered in project prose require verification: "
                + ", ".join(inferred_skills)
            )

        evidence_data = project.model_dump(exclude={"source_pages"}, exclude_none=True)
        evidence = ProjectEvidence(
            source="cv_project",
            reference=f"projects/{index}",
            text=json.dumps(evidence_data, ensure_ascii=False, separators=(",", ":")),
            source_pages=project.source_pages,
        )
        projects.append(StandardizedProject(
            title=_clean(project.title),
            company=_clean(project.company),
            role=_clean(project.role),
            start_date=_clean(project.start_date),
            end_date=_clean(project.end_date),
            description=_clean(project.description),
            tasks=_unique(project.tasks),
            skills=skills,
            source_pages=project.source_pages,
            evidence=[evidence],
            review_required=bool(reasons),
            review_reasons=reasons,
        ))
    return StandardizedProjects(projects=projects)
