import os
from langgraph.graph import StateGraph, END
from app.state import PipelineState
from app.nodes import ConditionExtractionNode, HCCEvaluationNode
from services.vertex_client import VertexLLMClient
from services.hcc_lookup import HCCLookupService
from config.settings import get_settings


def build_graph(extract_node, hcc_node):

    graph = StateGraph(PipelineState)
    graph.add_node("extract", extract_node)
    graph.add_node("hcc_eval", hcc_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "hcc_eval")
    graph.add_edge("hcc_eval", END)

    return graph.compile()


# 👇 Dev UI–compatible wrapper
def _build_default_graph():
    settings = get_settings()
    hcc = HCCLookupService(settings.hcc_csv_path)

    # Initialize Vertex client only if creds are configured; otherwise fall back to mock
    llm_client = None
    if settings.gcp_project and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            llm_client = VertexLLMClient(settings.gcp_project, settings.gcp_location)
        except Exception:
            # Safe fallback for local dev without valid GCP creds
            llm_client = None

    extract_node = ConditionExtractionNode(llm_client)
    hcc_node = HCCEvaluationNode(hcc)

    return build_graph(extract_node, hcc_node)

# 👇 THIS is what LangGraph Dev UI looks for
graph = _build_default_graph()