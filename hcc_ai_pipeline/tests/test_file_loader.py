from pathlib import Path
from src.ingestion.file_loader import FileLoader


def test_load_txt_files(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("Assessment/Plan: Diabetes")

    loader = FileLoader(str(tmp_path))
    notes = loader.load_files()

    assert "note.txt" in notes
