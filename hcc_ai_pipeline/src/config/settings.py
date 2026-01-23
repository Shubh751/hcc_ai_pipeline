from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    gcp_project: str
    gcp_location: str = "us-central1"
    gcp_credentials_path: str
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    hcc_csv_path: str = "data/hcc_codes.csv"


def get_settings() -> Settings:
    return Settings(
        gcp_project=os.getenv("GCP_PROJECT", "hcc_ai_pipeline"),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        gcp_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        input_dir=os.getenv("INPUT_DIR", "data/input"),
        output_dir=os.getenv("OUTPUT_DIR", "data/output"),
        hcc_csv_path=os.getenv("HCC_CSV_PATH", "data/hcc_codes.csv"),
    )
