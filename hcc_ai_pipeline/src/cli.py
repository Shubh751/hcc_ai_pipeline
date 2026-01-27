import json
import traceback
from config.settings import get_settings
from ingestion.file_loader import FileLoader
from services.hcc_lookup import HCCLookupService
from services.vertex_client import VertexLLMClient
from app.nodes import ConditionExtractionNode, HCCEvaluationNode
from app.graph import build_graph
from app.state import PipelineState


def main():
    try:
        settings = get_settings()

        loader = FileLoader(settings.input_dir)
        notes = loader.load_files()

        hcc_service = HCCLookupService(settings.hcc_csv_path)

        llm_client = None
        if settings.gcp_project:
            llm_client = VertexLLMClient(settings.gcp_project, settings.gcp_location)

        extract_node = ConditionExtractionNode(llm_client)
        hcc_node = HCCEvaluationNode(hcc_service)
        graph = build_graph(extract_node, hcc_node)

        for filename, text in notes.items():
            state = PipelineState(filename=filename, raw_text=text)
            result = graph.invoke(state)

            output_path = f"{settings.output_dir}/{filename}.json"
            with open(output_path, "w") as f:
                # LangGraph returns a dict, convert to PipelineState for validation and serialization
                if isinstance(result, dict):
                    result_state = PipelineState(**result)
                else:
                    result_state = result
                    
                # existing
                # json.dump(result_state.model_dump(), f, indent=2)
                # replace with:
                data = result_state.model_dump()
                # Down-convert extracted_conditions to a list of names for output
                data["extracted_conditions"] = [c.name for c in result_state.extracted_conditions]
                json.dump(data, f, indent=2)
                
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
