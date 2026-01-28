from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse

from config.settings import get_settings
from ingestion.file_loader import FileLoader
from services.hcc_lookup import HCCLookupService
from services.vertex_client import VertexLLMClient
from app.nodes import ConditionExtractionNode, HCCEvaluationNode
from app.graph import build_graph
from app.state import PipelineState

# ---------- Security ----------
security = HTTPBasic()

DEMO_USER = os.getenv("DEMO_USERNAME", "tushar_reviewer")
DEMO_PASS = os.getenv("DEMO_PASSWORD", "HccDemo2024!")


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != DEMO_USER or credentials.password != DEMO_PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------- App ----------
app = FastAPI(title="HCC AI Pipeline API", version="1.0.0")

WORKERS = int(os.getenv("WORKERS", "8"))
EXECUTOR = ThreadPoolExecutor(max_workers=WORKERS)


# ---------- Job Store ----------
class JobState:
    def __init__(self, files: List[str]):
        self.id: str = str(uuid.uuid4())
        self.status: str = "pending"
        self.submitted_at: str = self._now()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.total: int = len(files)
        self.completed: int = 0
        self.errors: List[str] = []
        self.outputs: List[str] = []
        self.files = files

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def _parse(ts: str) -> datetime:
        return datetime.fromisoformat(ts.replace("Z", ""))

    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            start = self._parse(self.started_at)
            end = self._parse(self.finished_at)
            return round((end - start).total_seconds(), 2)
        return None

    def elapsed_seconds(self) -> Optional[int]:
        if self.started_at and not self.finished_at:
            start = self._parse(self.started_at)
            return int((datetime.utcnow() - start).total_seconds())
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds(),
            "elapsed_seconds": self.elapsed_seconds(),
            "total": self.total,
            "completed": self.completed,
            "errors": self.errors,
            "outputs": self.outputs,
        }


JOB_STORE: Dict[str, JobState] = {}


# ---------- File Processing ----------
def _process_one(filename: str, text: str, graph, output_dir: str) -> str:
    state = PipelineState(filename=filename, raw_text=text)
    result = graph.invoke(state)

    result_state = PipelineState(**result) if isinstance(result, dict) else result

    data = result_state.model_dump()
    data["extracted_conditions"] = [c.name for c in result_state.extracted_conditions]

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{filename}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


# ---------- Background Job Runner ----------
def _run_job(job: JobState, settings):
    try:
        job.status = "running"
        job.started_at = job._now()

        loader = FileLoader(settings.input_dir)
        notes = loader.load_files()
        notes = {k: v for k, v in notes.items() if k in job.files}

        hcc_service = HCCLookupService(settings.hcc_csv_path)
        llm_client = None

        if settings.gcp_project:
            llm_client = VertexLLMClient(
                settings.gcp_project,
                settings.gcp_location,
            )

        graph = build_graph(
            ConditionExtractionNode(llm_client),
            HCCEvaluationNode(hcc_service),
        )

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = [
                ex.submit(_process_one, f, t, graph, settings.output_dir)
                for f, t in notes.items()
            ]

            for fut in as_completed(futures):
                try:
                    job.outputs.append(fut.result())
                    job.completed += 1
                except Exception as e:
                    job.errors.append(str(e))

        job.status = "completed" if not job.errors else "failed"
        job.finished_at = job._now()

    except Exception as e:
        job.errors.append(str(e))
        job.status = "failed"
        job.finished_at = job._now()


# ---------- API ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(_: str = Depends(authenticate)):
    return {"status": "ok"}


@app.post("/jobs")
def create_job(_: str = Depends(authenticate)):
    settings = get_settings()
    loader = FileLoader(settings.input_dir)
    notes = loader.load_files()

    if not notes:
        raise HTTPException(status_code=400, detail="No input files found")

    job = JobState(list(notes.keys()))
    JOB_STORE[job.id] = job

    EXECUTOR.submit(_run_job, job, settings)

    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _: str = Depends(authenticate)):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# ---------- UI ----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>HCC AI Pipeline</title>
  <style>
    body { font-family: system-ui; margin: 24px; }
    button { padding: 8px 14px; }
    pre { background: #f4f4f4; padding: 12px; }
  </style>
</head>
<body>
<h1>HCC AI Pipeline</h1>
<button id="start">Start Batch</button>
<pre id="status"></pre>

<script>
const statusEl = document.getElementById('status');
document.getElementById('start').onclick = async () => {
  statusEl.textContent = 'Starting job...';
  const r = await fetch('/jobs', { method: 'POST' });
  const d = await r.json();
  poll(d.job_id);
};

function poll(jobId) {
  const t = setInterval(async () => {
    const r = await fetch('/jobs/' + jobId);
    const d = await r.json();

    statusEl.textContent =
      'Status: ' + d.status +
      '\\nCompleted: ' + d.completed + '/' + d.total +
      '\\nElapsed: ' + (d.elapsed_seconds ?? '-') + ' sec' +
      '\\nDuration: ' + (d.duration_seconds ?? '-') + ' sec';

    if (d.status === 'completed' || d.status === 'failed') {
      clearInterval(t);
      statusEl.textContent += '\\n\\nOutputs:\\n' + (d.outputs || []).join('\\n');
    }
  }, 3000);
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index(_: str = Depends(authenticate)):
    return INDEX_HTML
