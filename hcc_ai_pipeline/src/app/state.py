from pydantic import BaseModel
from typing import List


class Condition(BaseModel):
    name: str
    code: str | None = None
    hcc_relevant: bool = False


class PipelineState(BaseModel):
    filename: str
    raw_text: str
    extracted_conditions: List[str] = []
    enriched_conditions: List[Condition] = []
