
from schemas import RawCV, Skill, Proficiency

cv = RawCV(
    skills=[
        Skill(
            name="Python",
            proficiency=Proficiency(
                type="stars",
                raw_value="4 out of 5 stars",
                value=4,
                maximum=5
            ),
            source_pages=[1]
        )
    ]
)

print(cv.model_dump_json(indent=2))
