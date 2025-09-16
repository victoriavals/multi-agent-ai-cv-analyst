from pydantic import BaseModel, Field
from typing import Any, Optional


class CVStruct(BaseModel):
    basics: dict | None = None
    skills: list[str] = []
    experience: list[dict] = []
    projects: list[dict] = []
    education: list[dict] = []
    certifications: list[dict] = []


class SkillProfile(BaseModel):
    explicit: list[str]
    implicit: list[str]
    transferable: list[str]
    seniority_signals: list[str]
    coverage_map: dict = Field(default_factory=dict)  # {market_skill: {match: float, evidence: list[str]}}


class MarketSummary(BaseModel):
    role: str
    region: str
    in_demand_skills: list[str]
    common_tools: list[str]
    frameworks: list[str]
    nice_to_have: list[str]
    sources_sample: list[str] = []


class GraphState(BaseModel):
    file_path: str | None = None
    cv_text: str | None = None
    cv_struct: CVStruct | None = None
    target_role: str = "Senior AI Engineer"
    market_region: str = "Global"
    skill_profile: SkillProfile | None = None
    market_summary: MarketSummary | None = None
    report_md: str | None = None
    output_path: str | None = None
    lang: str = "en"
    logs: list[str] = []
    error: str | None = None
    retry_count: int = 0
    # Provider configuration
    provider: str = "auto"
    chat_model_name: str | None = None
    embed_model_name: str | None = None
