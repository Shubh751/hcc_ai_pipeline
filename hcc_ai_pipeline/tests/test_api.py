import pytest
import json
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ensure src is on the path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from server.api import app, JobState, JOB_STORE, DEMO_USER, DEMO_PASS
from app.state import PipelineState
from services.hcc_lookup import HCCLookupService
from ingestion.file_loader import FileLoader

client = TestClient(app)

# Helper: override settings for tests
@pytest.fixture
def mock_settings():
    with patch('server.api.get_settings') as m:
        m.return_value = MagicMock(
            input_dir='tests/fixtures/input',
            output_dir='tests/fixtures/output',
            hcc_csv_path='data/HCC_relevant_codes.csv',
            gcp_project='test-project',
            gcp_location='us-central1'
        )
        yield m

@pytest.fixture
def create_input_files(tmp_path):
    inp = tmp_path / 'input'
    inp.mkdir()
    (inp / 'note1.txt').write_text('Assessment: Diabetes')
    (inp / 'note2.txt').write_text('Assessment: Hypertension')
    out = tmp_path / 'output'
    out.mkdir()
    return str(inp), str(out)

# ---------- Authentication ----------
def test_health_no_auth():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok'}

def test_login_success():
    r = client.post('/login', auth=(DEMO_USER, DEMO_PASS))
    assert r.status_code == 200
    assert r.json() == {'status': 'ok'}

def test_login_failure():
    r = client.post('/login', auth=('bad', 'creds'))
    assert r.status_code == 401
    assert 'Invalid credentials' in r.json()['detail']

def test_protected_route_without_auth():
    r = client.get('/')
    assert r.status_code == 401

# ---------- Job Creation ----------
def test_create_job_no_files(mock_settings):
    with patch('server.api.FileLoader') as MockLoader:
        MockLoader.return_value.load_files.return_value = {}
        r = client.post('/jobs', auth=(DEMO_USER, DEMO_PASS))
        assert r.status_code == 400
        assert 'No input files found' in r.json()['detail']

def test_create_job_success(mock_settings, create_input_files):
    inp_dir, out_dir = create_input_files
    mock_settings.return_value.input_dir = inp_dir
    mock_settings.return_value.output_dir = out_dir

    with patch('server.api.FileLoader') as MockLoader:
        MockLoader.return_value.load_files.return_value = {'note1.txt': 'Assessment: Diabetes'}
        r = client.post('/jobs', auth=(DEMO_USER, DEMO_PASS))
        assert r.status_code == 200
        job_id = r.json()['job_id']
        assert job_id in JOB_STORE
        assert JOB_STORE[job_id].status in {'pending', 'running'}

# ---------- Job Status ----------
def test_get_job_status_not_found():
    r = client.get('/jobs/fake-id', auth=(DEMO_USER, DEMO_PASS))
    assert r.status_code == 404

def test_get_job_status_success(mock_settings, create_input_files):
    inp_dir, out_dir = create_input_files
    mock_settings.return_value.input_dir = inp_dir
    mock_settings.return_value.output_dir = out_dir

    with patch('server.api.FileLoader') as MockLoader:
        MockLoader.return_value.load_files.return_value = {'note1.txt': 'Assessment: Diabetes'}
        # Create a job manually
        job = JobState(['note1.txt'])
        JOB_STORE[job.id] = job
        r = client.get(f'/jobs/{job.id}', auth=(DEMO_USER, DEMO_PASS))
        assert r.status_code == 200
        data = r.json()
        assert data['job_id'] == job.id
        assert data['status'] == 'pending'
        assert data['total'] == 1
        assert data['completed'] == 0
        assert isinstance(data['submitted_at'], str)

# ---------- Background Job Execution ----------
def test_run_job_success(mock_settings, create_input_files):
    inp_dir, out_dir = create_input_files
    mock_settings.return_value.input_dir = inp_dir
    mock_settings.return_value.output_dir = out_dir
    mock_settings.return_value.gcp_project = None  # Disable LLM for simplicity

    # Create a job
    job = JobState(['note1.txt'])
    JOB_STORE[job.id] = job

    # Mock graph and processing
    with patch('server.api.build_graph') as mock_graph:
        mock_graph.return_value.invoke.return_value = PipelineState(
            filename='note1.txt',
            raw_text='Assessment: Diabetes',
            extracted_conditions=[],
            enriched_conditions=[]
        )
        # Run the job
        from server.api import _run_job
        _run_job(job, mock_settings.return_value)

        assert job.status == 'completed'
        assert job.completed == 1
        assert job.finished_at is not None
        assert len(job.outputs) == 1
        # Verify output file exists and is valid JSON
        assert os.path.exists(job.outputs[0])
        with open(job.outputs[0]) as f:
            data = json.load(f)
            assert 'filename' in data
            assert 'extracted_conditions' in data

def test_run_job_with_errors(mock_settings, create_input_files):
    inp_dir, out_dir = create_input_files
    mock_settings.return_value.input_dir = inp_dir
    mock_settings.return_value.output_dir = out_dir
    mock_settings.return_value.gcp_project = None

    job = JobState(['note1.txt'])
    JOB_STORE[job.id] = job

    # Force an exception during processing
    with patch('server.api.build_graph') as mock_graph:
        mock_graph.return_value.invoke.side_effect = RuntimeError('Boom')
        from server.api import _run_job
        _run_job(job, mock_settings.return_value)

        assert job.status == 'failed'
        assert 'Boom' in job.errors
        assert job.finished_at is not None

# ---------- JobState Helpers ----------
def test_job_state_duration_and_elapsed():
    job = JobState(['a.txt'])
    # Simulate timing
    now = datetime.utcnow()
    job.started_at = (now.isoformat() + 'Z')
    # No finish yet
    assert job.duration_seconds() is None
    assert job.elapsed_seconds() is not None and isinstance(job.elapsed_seconds(), int)
    # Finish
    job.finished_at = (now.isoformat() + 'Z')
    assert job.duration_seconds() == 0.0
    assert job.elapsed_seconds() is None

def test_job_state_to_dict():
    job = JobState(['a.txt'])
    d = job.to_dict()
    expected_keys = {
        'job_id', 'status', 'submitted_at', 'started_at', 'finished_at',
        'duration_seconds', 'elapsed_seconds', 'total', 'completed',
        'errors', 'outputs'
    }
    assert set(d.keys()) == expected_keys
    assert d['job_id'] == job.id
    assert d['status'] == 'pending'
    assert d['total'] == 1
    assert d['completed'] == 0

# ---------- UI Rendering ----------
def test_index_html_requires_auth():
    r = client.get('/')
    assert r.status_code == 401

def test_index_html_success():
    r = client.get('/', auth=(DEMO_USER, DEMO_PASS))
    assert r.status_code == 200
    assert '<h1>HCC AI Pipeline</h1>' in r.text
    assert 'Start Batch' in r.text
    assert '<script>' in r.text
