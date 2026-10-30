"""Project transformation contract tests; no model inference required."""
import unittest

from pydantic import ValidationError

from components.vlm_core_data_extraction.schemas import Project, RawCV, Skill
from components.llm_core_data_transformation.projects_handler import transform_projects
from components.llm_core_data_transformation.schemas import StandardizedProject, StandardizedSkills
from components.llm_core_data_transformation.skills_handler import transform_skills


class ProjectsTransformationTests(unittest.TestCase):
    def test_renderer_fields_and_evidence_are_preserved(self):
        raw = RawCV(projects=[Project(
            title="Payments platform",
            company="Example Client",
            role="Backend Developer",
            start_date="01/2024",
            end_date="Present",
            description="Built payment services with Python and Docker.",
            tasks=["Implemented APIs", "Implemented APIs", "  Supported releases  "],
            skills=["Python", "Docker"],
            source_pages=[2, 3],
        )])
        project = transform_projects(raw).projects[0]
        self.assertEqual(project.company, "Example Client")
        self.assertEqual(project.tasks, ["Implemented APIs", "Supported releases"])
        self.assertEqual(project.skills, ["Python", "Docker"])
        self.assertEqual(project.source_pages, [2, 3])
        self.assertEqual(project.evidence[0].reference, "projects/0")
        self.assertFalse(project.review_required)

    def test_prose_skill_discovery_is_conservative_and_reviewable(self):
        raw = RawCV(
            skills=[Skill(name="Go")],
            projects=[Project(
                title="Inventory",
                role="Developer",
                description="Maintained Go and TypeScript services.",
                tasks=["Delivered backend changes"],
            )],
        )
        project = transform_projects(raw).projects[0]
        self.assertEqual(project.skills, ["TypeScript", "Go"])
        self.assertTrue(project.review_required)
        self.assertIn("require verification", project.review_reasons[-1])

    def test_missing_content_is_not_invented_and_is_flagged(self):
        project = transform_projects(RawCV(projects=[Project(title="Student project")])).projects[0]
        self.assertIsNone(project.role)
        self.assertEqual(project.tasks, [])
        self.assertEqual(project.skills, [])
        self.assertTrue(project.review_required)
        self.assertIn("Missing project role", project.review_reasons)

    def test_blank_tasks_are_cleaned_and_flagged_as_missing(self):
        project = transform_projects(RawCV(projects=[Project(
            title="Project", role="Developer", description="Built a service.",
            tasks=["   ", "\t"], skills=["Python"],
        )])).projects[0]
        self.assertEqual(project.tasks, [])
        self.assertTrue(project.review_required)
        self.assertIn("Missing project tasks", project.review_reasons)

    def test_projects_are_in_combined_renderer_output(self):
        result = transform_skills(RawCV(projects=[Project(title="Project")]))
        self.assertEqual(result.projects[0].title, "Project")
        loaded = StandardizedSkills.model_validate_json(result.model_dump_json())
        self.assertEqual(loaded.projects[0].title, "Project")

    def test_flagged_project_cannot_disable_review(self):
        with self.assertRaises(ValidationError):
            StandardizedProject(
                title="Project", review_required=False, review_reasons=["Check this"],
            )


if __name__ == "__main__":
    unittest.main()
