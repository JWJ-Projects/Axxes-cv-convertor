"""Proficiency resolution plus the existing model evaluation utilities."""
from pathlib import Path
from typing import Callable

from .proficiency_format import format_proficiency_input
from .schemas import LABELS, StandardizedProficiency, StandardizedSkill

COMPONENT_DIR = Path(__file__).resolve().parent

# A predictor receives a canonical skill name and only its collected evidence.
ProficiencyPredictor = Callable[[str, list[str]], str | None]
class Sequence_Evaluator():
    def __init__(self):
        from gliner2 import AutoExtractor
        import pandas as pd

        self.model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
        self.test_data = pd.read_csv(COMPONENT_DIR / "data" / "developer_proficiency_test.csv")
        return
    def load_stage1(self):
        self.model.load_adapter(str(COMPONENT_DIR / "decide_proficiency_finetuned_stage1" / "best"))
        return
    def evaluate(self, dataset=None):
        import numpy as np

        if dataset is not None:
            df = dataset
        else:
            df = self.test_data
        conf_matrix = np.zeros((4,4)) # Using a 4 by 4 confusion matrix to capture the results
        labels = ["Junior", "Confirmed", "Advanced", "Expert"]
        for i, row in df.iterrows():
            proficiency_test = self.model.classify_text(
                format_proficiency_input(
                    row["language/framework/tool"], [row["description"]],
                ),
                {"proficiency": [
                    "Junior", "Confirmed", "Advanced", "Expert"
                ]}
            )
            conf_matrix[labels.index(proficiency_test["proficiency"]), labels.index(row["proficiency"])] += 1
        return conf_matrix
class Sequence_Metrics():
    def __init__(self, conf_matrix):
        self.conf_matrix = conf_matrix
        return
    def accuracy(self):
        correct = 0
        for i in range(0,4):
            correct += self.conf_matrix[i,i]
        return correct / self.conf_matrix.sum()
    def recall(self, label: int = 0):
        tp = self.conf_matrix[label, label]
        y_label_true = self.conf_matrix[:, label].sum()
        return tp / y_label_true
    def precision(self, label: int = 0):
        tp = self.conf_matrix[label, label]
        y_label_pred = self.conf_matrix[label, :].sum()
        return tp / y_label_pred
def main():
    seq_eval = Sequence_Evaluator()
    seq_eval.load_stage1()
    conf_matrix = seq_eval.evaluate()
    metrics = Sequence_Metrics(conf_matrix)
    print(f"Accuracy: {metrics.accuracy()}")
    print(f"Recall: {metrics.recall()}")
    print(f"Precision: {metrics.precision()}")
    print(conf_matrix)
    return
def predict(skill: StandardizedSkill, predictor: ProficiencyPredictor | None = None) -> StandardizedSkill:
    """Resolve a skill without mutating the collected evidence."""
    result = skill.model_copy(deep=True)
    result.proficiency = None
    reasons: list[str] = []
    explicit: list[StandardizedProficiency] = []
    for evidence in skill.evidence:
        raw = evidence.original_proficiency
        if not raw:
            continue
        raw_text = str(raw.get("raw_value") or "").strip().casefold()
        label = next((label for label in LABELS if label.casefold() == raw_text), None)
        numeric_level = None
        value, maximum = raw.get("value"), raw.get("maximum")
        if maximum == 4 and isinstance(value, (int, float)) and value in (1, 2, 3, 4):
            numeric_level = int(value)
        if label and numeric_level and LABELS[numeric_level - 1] != label:
            reasons.append("Conflicting label and numeric rating in source evidence")
        if label:
            explicit.append(StandardizedProficiency(
                label=label, level=LABELS.index(label) + 1, method="explicit_label",
            ))
        elif numeric_level:
            explicit.append(StandardizedProficiency(
                label=LABELS[numeric_level - 1], level=numeric_level,
                method="explicit_four_point_scale",
            ))
        elif any(v is not None for v in raw.values()):
            reasons.append("Original proficiency scale needs an approved conversion")

    levels = {p.level for p in explicit}
    if len(levels) > 1:
        reasons.append("Conflicting explicit proficiency ratings")
    elif explicit and not reasons:
        result.proficiency = explicit[0]
    elif not explicit and not reasons and predictor is not None:
        # A bare technology list is not evidence of a proficiency level.
        texts = list(dict.fromkeys(
            e.text for e in skill.evidence
            if e.source in ("cv_project", "interview") and len(e.text.split()) >= 5
        ))
        if texts:
            label = predictor(skill.name, texts)
            if label in LABELS:
                result.proficiency = StandardizedProficiency(
                    label=label, level=LABELS.index(label) + 1, method="model_inference",
                )
                reasons.append("Model-inferred proficiency requires human confirmation")
            elif label is not None:
                reasons.append("Predictor returned an unsupported proficiency label")
    if result.proficiency is None and not reasons:
        reasons.append("Insufficient evidence for a proficiency rating")
    result.review_reasons = list(dict.fromkeys(reasons))
    result.review_required = result.proficiency is None or bool(reasons)
    return result


class GLiNERProficiencyPredictor:
    """Opt-in lazy model loading; never downloads a model during transformation by default."""
    def __init__(self, adapter_path: str | Path):
        self.adapter_path = Path(adapter_path)
        self.model = None

    def __call__(self, skill_name: str, evidence: list[str]) -> str | None:
        if self.model is None:
            if not self.adapter_path.is_dir():
                raise FileNotFoundError(f"Missing proficiency adapter: {self.adapter_path}")
            from gliner2 import AutoExtractor
            model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
            model.load_adapter(str(self.adapter_path))
            self.model = model
        text = format_proficiency_input(skill_name, evidence)
        response = self.model.classify_text(text, {"proficiency": list(LABELS)})
        return response.get("proficiency")
if __name__ == '__main__':
    main()