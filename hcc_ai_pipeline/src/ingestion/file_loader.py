from pathlib import Path
from typing import Dict
from loguru import logger
import fitz  # PyMuPDF for PDF
from docx import Document


class FileLoader:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

    def load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file using PyMuPDF."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def load_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file using python-docx."""
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text

    def load_files(self) -> Dict[str, str]:
        """Load TXT, PDF, and DOCX files from the input directory."""
        notes: Dict[str, str] = {}

        files = list(self.input_dir.iterdir())
        logger.info(f"Found {len(files)} files in {self.input_dir}")

        for file_path in files:
            try:
                if file_path.suffix.lower() == ".txt" or file_path.suffix.lower() == "":
                    content = file_path.read_text(encoding="utf-8")
                elif file_path.suffix.lower() == ".pdf":
                    content = self.load_pdf(file_path)
                elif file_path.suffix.lower() == ".docx":
                    content = self.load_docx(file_path)
                else:
                    logger.warning(f"Skipping unsupported file: {file_path.name}")
                    continue

                notes[file_path.name] = content.strip()
                logger.debug(f"Loaded: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Successfully loaded {len(notes)} files")
        return notes
