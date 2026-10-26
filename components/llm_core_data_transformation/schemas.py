"""Contract between skill transformation and CV rendering (no layout decisions)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProficiencyLabel = Literal["Junior", "Confirmed", "Advanced", "Expert"]
LABELS: tuple[ProficiencyLabel, ...] = ("Junior", "Confirmed", "Advanced", "Expert")


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkillEvidence(ContractModel):
    source: Literal["cv_skill", "cv_project", "interview"]
    reference: str
    text: str
    source_pages: list[int] = Field(default_factory=list)
    # Retain the original scale rather than overwriting it with a prediction.
    original_proficiency: dict | None = None


class StandardizedProficiency(ContractModel):
    label: ProficiencyLabel
    level: int = Field(ge=1, le=4)
    maximum: Literal[4] = 4
    method: Literal["explicit_label", "explicit_four_point_scale", "model_inference"]

    @model_validator(mode="after")
    def consistent_level(self):
        if LABELS[self.level - 1] != self.label:
            raise ValueError("Proficiency label must match its four-bar level")
        return self


class StandardizedSkill(ContractModel):
    name: str = Field(min_length=1)
    category: str | None = None
    proficiency: StandardizedProficiency | None = None
    evidence: list[SkillEvidence] = Field(default_factory=list)
    review_required: bool = True
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_review(self):
        inferred = self.proficiency is not None and self.proficiency.method == "model_inference"
        if self.proficiency is None or self.review_reasons or inferred:
            if not self.review_required:
                raise ValueError("Unknown or flagged proficiency requires review")
        return self


class StandardizedSkills(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    skills: list[StandardizedSkill] = Field(default_factory=list)
