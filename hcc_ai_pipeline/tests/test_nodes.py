from src.app.nodes import ConditionExtractionNode, HCCEvaluationNode
from src.app.state import PipelineState
from src.services.hcc_lookup import HCCLookupService
import pandas as pd


def test_nodes(tmp_path):
    csv = tmp_path / "hcc.csv"
    pd.DataFrame(
        [{"Description": "Diabetes Mellitus", "ICD-10-CM Codes": "E11", "Tags": "True"}]
    ).to_csv(csv, index=False)

    state = PipelineState(filename="a.txt", raw_text="Assessment/Plan: Diabetes")
    hcc = HCCLookupService(str(csv))

    extract = ConditionExtractionNode(llm_client=None)
    eval_node = HCCEvaluationNode(hcc)

    state = extract(state)
    state = eval_node(state)

    assert state.enriched_conditions[0].code == "E11"
