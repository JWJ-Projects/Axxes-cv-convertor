"""Offline tests: no torch, model downloads, training, or source-data changes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from pydantic import ValidationError

from components.vlm_core_data_extraction.schemas import RawCV, Skill, Proficiency, Project
from components.llm_core_data_transformation.proficiency_resolver import (
    GLiNERProficiencyPredictor, Sequence_Evaluator, format_proficiency_input,
)
from components.llm_core_data_transformation.schemas import StandardizedSkills, StandardizedProficiency
from components.llm_core_data_transformation.skill_normalizer import SkillNormalizer
from components.llm_core_data_transformation.skills_handler import transform_skills


class SkillsTransformationTests(unittest.TestCase):
    def transform(self, *skills, **kwargs):
        return transform_skills(RawCV(skills=list(skills)), **kwargs)

    def test_all_four_labels(self):
        labels = ("Junior", "Confirmed", "Advanced", "Expert")
        for level, label in enumerate(labels, 1):
            skill = self.transform(Skill(name="Python", proficiency=Proficiency(
                type="text", raw_value=label,
            ))).skills[0]
            self.assertEqual(skill.proficiency.level, level)
            self.assertFalse(skill.review_required)

    def test_four_point_scale(self):
        skill = self.transform(Skill(name="Java", proficiency=Proficiency(
            type="bars", value=3, maximum=4,
        ))).skills[0]
        self.assertEqual(skill.proficiency.label, "Advanced")

    def test_other_scale_not_guessed(self):
        skill = self.transform(Skill(name="Python", proficiency=Proficiency(
            type="stars", value=4, maximum=5,
        ))).skills[0]
        self.assertIsNone(skill.proficiency)
        self.assertTrue(skill.review_required)
        self.assertEqual(skill.evidence[0].original_proficiency["maximum"], 5)

    def test_alias_merge_and_unknown_preservation(self):
        result = self.transform(Skill(name="TS"), Skill(name="TypeScript"), Skill(name="MyCustomTool"))
        self.assertEqual([s.name for s in result.skills], ["TypeScript", "MyCustomTool"])
        self.assertEqual(len(result.skills[0].evidence), 2)
        self.assertIsNone(result.skills[1].category)

    def test_related_technology_is_not_an_alias(self):
        result = self.transform(Skill(name="Spring MVC"), Skill(name="Spring Boot"))
        self.assertEqual(len(result.skills), 2)

    def test_reviewed_duplicate_taxonomy_category_is_deterministic(self):
        result = self.transform(Skill(name="Mulesoft"))
        self.assertEqual(result.skills[0].category, "ERP/CRM & Integration")

    def test_unreviewed_conflicting_taxonomy_categories_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            taxonomy = Path(tmp) / "taxonomy.csv"
            taxonomy.write_text(
                "Category,Skill\nFirst,Duplicate\nSecond,Duplicate\n", encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting taxonomy categories"):
                SkillNormalizer(taxonomy)

    def test_conflicting_explicit_values(self):
        result = self.transform(*[
            Skill(name="Python", proficiency=Proficiency(type="text", raw_value=label))
            for label in ("Junior", "Expert")
        ])
        self.assertIsNone(result.skills[0].proficiency)
        self.assertIn("Conflicting explicit proficiency ratings", result.skills[0].review_reasons)

    def test_conflicting_label_and_bars(self):
        skill = self.transform(Skill(name="Java", proficiency=Proficiency(
            type="bars", raw_value="Junior", value=4, maximum=4,
        ))).skills[0]
        self.assertIsNone(skill.proficiency)
        self.assertTrue(skill.review_required)

    def test_project_and_interview_evidence(self):
        raw = RawCV(projects=[Project(
            skills=["Go"], description="Developed Python services for our inventory system.",
            source_pages=[2],
        )])
        result = transform_skills(raw, "I maintain TypeScript applications for our customers.")
        by_name = {s.name: s for s in result.skills}
        self.assertIn("Go", by_name)
        self.assertIn("Python", by_name)
        self.assertIn("TypeScript", by_name)
        self.assertEqual(by_name["Python"].evidence[0].source_pages, [2])
        self.assertTrue(all(s.proficiency is None for s in result.skills))

    def test_bare_mentions_do_not_call_predictor(self):
        def fail(*args):
            self.fail("A bare skill must not trigger inference")
        result = self.transform(Skill(name="Python"), predictor=fail)
        self.assertIsNone(result.skills[0].proficiency)

    def test_inference_is_targeted_and_flagged(self):
        calls = []
        def predictor(name, evidence):
            calls.append((name, evidence))
            return "Confirmed"
        result = self.transform(
            Skill(name="Python"), Skill(name="Java"),
            interview_transcript="I maintain Python services independently for our customers.",
            predictor=predictor,
        )
        self.assertEqual([name for name, _ in calls], ["Python"])
        self.assertEqual(result.skills[0].proficiency.level, 2)
        self.assertTrue(result.skills[0].review_required)
        self.assertIsNone(result.skills[1].proficiency)

    def test_invalid_prediction_stays_unknown(self):
        result = self.transform(
            Skill(name="Python"),
            interview_transcript="I maintain Python services independently for our customers.",
            predictor=lambda *_: "Master",
        )
        self.assertIsNone(result.skills[0].proficiency)

    def test_production_predictor_uses_shared_training_format(self):
        predictor = GLiNERProficiencyPredictor("unused")
        predictor.model = MagicMock()
        predictor.model.classify_text.return_value = {"proficiency": "Advanced"}
        evidence = ["I maintain services independently.", "I review complex changes."]
        self.assertEqual(predictor("Python", evidence), "Advanced")
        predictor.model.classify_text.assert_called_once_with(
            format_proficiency_input("Python", evidence),
            {"proficiency": ["Junior", "Confirmed", "Advanced", "Expert"]},
        )

    def test_evaluator_uses_production_input_format(self):
        import pandas as pd

        evaluator = Sequence_Evaluator.__new__(Sequence_Evaluator)
        evaluator.model = MagicMock()
        evaluator.model.classify_text.return_value = {"proficiency": "Junior"}
        dataset = pd.DataFrame([{
            "description": "I am learning the basics.",
            "language/framework/tool": "Python",
            "proficiency": "Junior",
        }])
        matrix = evaluator.evaluate(dataset)
        self.assertEqual(matrix[0, 0], 1)
        evaluator.model.classify_text.assert_called_once_with(
            format_proficiency_input("Python", ["I am learning the basics."]),
            {"proficiency": ["Junior", "Confirmed", "Advanced", "Expert"]},
        )

    def test_schema_and_reproducibility(self):
        raw = RawCV(skills=[Skill(name="Python")])
        first = transform_skills(raw).model_dump_json()
        self.assertEqual(first, transform_skills(raw).model_dump_json())
        self.assertEqual(StandardizedSkills.model_validate_json(first).skills[0].name, "Python")
        with self.assertRaises(ValidationError):
            StandardizedProficiency(label="Junior", level=4, method="explicit_label")


if __name__ == "__main__":
    unittest.main()
