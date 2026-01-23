from loguru import logger
from app.state import PipelineState, Condition
from ingestion.text_cleaner import extract_assessment_section
from services.hcc_lookup import HCCLookupService


class ConditionExtractionNode:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def __call__(self, state: PipelineState) -> PipelineState:
        logger.info(f"Extracting conditions from {state.filename}")
        section = extract_assessment_section(state.raw_text)

        if self.llm_client:
            conditions = self.llm_client.extract_conditions(section)
        else:
            conditions = ["Diabetes", "Hypertension"]

        state.extracted_conditions = conditions
        return state


class HCCEvaluationNode:
    def __init__(self, hcc_service: HCCLookupService):
        self.hcc_service = hcc_service

    def __call__(self, state: PipelineState) -> PipelineState:
        enriched = []

        for cond in state.extracted_conditions:
            match = self.hcc_service.lookup(cond)
            if match:
                enriched.append(
                    Condition(
                        name=match["condition"],
                        code=match["code"],
                        hcc_relevant=match["hcc_relevant"],
                    )
                )
            else:
                enriched.append(Condition(name=cond))

        state.enriched_conditions = enriched
        return state
