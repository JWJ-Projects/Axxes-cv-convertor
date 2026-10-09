
from pydantic import BaseModel, Field
from typing import Literal

class PersonalInformation(BaseModel):
    name: str | None = None
    current_occupation: str | None = None
    city: str | None = None
    country: str | None = None
    date_of_birth: str | None = None
    introduction: str | None = None


class Proficiency(BaseModel):
    type: Literal[
        "text",
        "stars",
        "bars",
        "percentage",
        "numeric",
        "other"
    ] | None = None

    raw_value: str | None = None
    value: float | None = None
    maximum: float | None = None


class Skill(BaseModel):
    name: str
    proficiency: Proficiency | None = None
    source_pages: list[int] = Field(default_factory=list)

class Project(BaseModel):
    company: str | None = None
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None

    tasks: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)

    source_pages: list[int] = Field(default_factory=list)

class Education(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    source_pages: list[int] = Field(default_factory=list)

class Certificate(BaseModel):
    name: str
    issuer: str | None = None
    date_achieved: str | None = None
    expiration_date: str | None = None

    source_pages: list[int] = Field(default_factory=list)

class Language(BaseModel):
    name: str
    understanding: str | None = None
    speaking: str | None = None
    writing: str | None = None
    overall_level: str | None = None

class Training(BaseModel):
    name: str
    organization: str | None = None
    date: str | None = None

class Weblink(BaseModel):
    label: str | None = None
    url: str

class RawCV(BaseModel):
    personal_information: PersonalInformation = Field(
        default_factory=PersonalInformation
    )

    skills: list[Skill] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certificates: list[Certificate] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    trainings: list[Training] = Field(default_factory=list)
    weblinks: list[Weblink] = Field(default_factory=list)

    additional_information: list[str] = Field(default_factory=list)
