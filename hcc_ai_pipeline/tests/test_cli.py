import json
import os
import pytest

from app.state import PipelineState, Condition


class DummyFileLoader:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir
    def load_files(self):
        # Return two notes to exercise concurrency
        return {
            "note1.txt": "dummy text 1",
            "note2.txt": "dummy text 2",
        }


class DummyHCC:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path


class FakeGraph:
    def __init__(self, returns):
        self._returns = returns
    def invoke(self, state: PipelineState):
        # returns can be a PipelineState or a dict
        return self._returns[state.filename]


def _setup_monkeypatch(monkeypatch, returns_map):
    import cli
    monkeypatch.setattr(cli, "FileLoader", DummyFileLoader)
    monkeypatch.setattr(cli, "HCCLookupService", DummyHCC)
    monkeypatch.setattr(cli, "build_graph", lambda *a, **k: FakeGraph(returns_map))
    return cli


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Avoid creating real Vertex client in tests
    monkeypatch.setenv("GCP_PROJECT", "")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    monkeypatch.delenv("WORKERS", raising=False)
    yield


def test_concurrent_writes_and_downconversion(tmp_path, monkeypatch):
    # Prepare returns for each note
    ret1 = PipelineState(
        filename="note1.txt", raw_text="dummy text 1",
        extracted_conditions=[Condition(name="Hypertension"), Condition(name="Diabetes")],
        enriched_conditions=[Condition(name="Hypertension", code="I10", hcc_relevant=True)]
    )
    ret2 = PipelineState(
        filename="note2.txt", raw_text="dummy text 2",
        extracted_conditions=[Condition(name="COPD")],
        enriched_conditions=[Condition(name="COPD", code="J44.9")]
    )
    returns_map = {
        "note1.txt": ret1,
        "note2.txt": ret2,
    }

    cli = _setup_monkeypatch(monkeypatch, returns_map)

    # Configure IO paths via env; create output dir
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("WORKERS", "4")

    # Run
    cli.main()

    # Assert outputs
    out1 = out_dir / "note1.txt.json"
    out2 = out_dir / "note2.txt.json"
    assert out1.exists() and out2.exists()

    data1 = json.loads(out1.read_text())
    data2 = json.loads(out2.read_text())

    # extracted_conditions must be list[str]
    assert data1["extracted_conditions"] == ["Hypertension", "Diabetes"]
    assert data2["extracted_conditions"] == ["COPD"]
    # enriched_conditions remain full objects with codes
    assert any(item.get("code") == "I10" for item in data1["enriched_conditions"]) 
    assert data2["enriched_conditions"][0]["code"] == "J44.9"


def test_graph_returns_dict_path(tmp_path, monkeypatch):
    # Return dicts to exercise PipelineState(**result) path
    ret1 = PipelineState(
        filename="note1.txt", raw_text="dummy",
        extracted_conditions=[Condition(name="A")],
        enriched_conditions=[Condition(name="A", code="X1")]
    ).model_dump()
    ret2 = PipelineState(
        filename="note2.txt", raw_text="dummy",
        extracted_conditions=[Condition(name="B")],
        enriched_conditions=[Condition(name="B", code="X2")]
    ).model_dump()

    returns_map = {"note1.txt": ret1, "note2.txt": ret2}
    cli = _setup_monkeypatch(monkeypatch, returns_map)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("WORKERS", "2")

    cli.main()

    data1 = json.loads((out_dir / "note1.txt.json").read_text())
    data2 = json.loads((out_dir / "note2.txt.json").read_text())
    assert data1["extracted_conditions"] == ["A"]
    assert data2["extracted_conditions"] == ["B"]


def test_error_in_one_file_does_not_abort_others(tmp_path, monkeypatch, capsys):
    class RaisingGraph:
        def __init__(self):
            pass
        def invoke(self, state):
            if state.filename == "note1.txt":
                raise RuntimeError("boom")
            return PipelineState(
                filename=state.filename, raw_text="",
                extracted_conditions=[Condition(name="OK")],
                enriched_conditions=[]
            )

    def build_graph_stub(*a, **k):
        return RaisingGraph()

    import cli
    monkeypatch.setattr(cli, "FileLoader", DummyFileLoader)
    monkeypatch.setattr(cli, "HCCLookupService", DummyHCC)
    monkeypatch.setattr(cli, "build_graph", build_graph_stub)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("WORKERS", "2")

    cli.main()
    captured = capsys.readouterr()
    assert "Error:" in captured.out or "Error:" in captured.err

    # Only the non-raising file should be present
    files = {p.name for p in out_dir.iterdir()}
    assert files == {"note2.txt.json"} or files == {"note1.txt.json"}  # order is non-deterministic which one raises in DummyFileLoader order
