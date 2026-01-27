from pydantic import BaseModel
from typing import List


class Condition(BaseModel):
    """Structured representation of a medical condition.
    Attributes:
        name: Canonical or display name of the condition.
        code: ICD-10-CM code if found in the lookup; otherwise None.
        hcc_relevant: True if the condition is marked HCC-relevant in the source CSV.
    """
    name: str
    code: str | None = None
    hcc_relevant: bool = False


class PipelineState(BaseModel):
    """State object carried through the LangGraph pipeline.
    Attributes:
        filename: Source filename of the note being processed.
        raw_text: Entire note text as ingested from the file.
        extracted_conditions: Structured `Condition` entries extracted from the note.
        enriched_conditions: Structured `Condition` entries post HCC/ICD enrichment.
    """
    filename: str
    raw_text: str
    extracted_conditions: List[Condition] = []
    enriched_conditions: List[Condition] = []
