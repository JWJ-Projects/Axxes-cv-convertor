"""Shared proficiency-classifier input formatting."""


def format_proficiency_input(skill_name: str, evidence: list[str]) -> str:
    """Build the exact input format shared by training, evaluation and inference."""
    return f"Target technology: {skill_name}\nEvidence:\n" + "\n".join(evidence)
