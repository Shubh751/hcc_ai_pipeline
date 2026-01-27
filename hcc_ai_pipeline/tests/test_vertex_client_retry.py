import types
import builtins
import pytest

from services.vertex_client import VertexLLMClient

class DummyModel:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)

    def generate_content(self, prompt):
        if not self._side_effects:
            raise AssertionError("No more side effects configured")
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return types.SimpleNamespace(text=effect)

@pytest.fixture
def client(monkeypatch):
    # Bypass real credential validation and vertex init
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", __file__)  # any file exists
    c = VertexLLMClient.__new__(VertexLLMClient)
    # Disable actual init steps
    c.model = None
    return c

def test_retry_then_success(monkeypatch, client):
    seq = [Exception("503 Service Unavailable"), Exception("resource_exhausted"), '["A","B"]']
    client.model = DummyModel(seq)

    sleeps = []
    monkeypatch.setattr("services.vertex_client.time.sleep", lambda s: sleeps.append(s))

    # Force should-retry True for our exceptions by message
    assert client._retry_generate("prompt", max_retries=5)
    # Expect at least two sleeps (for 2 retryable failures)
    assert len(sleeps) == 2
    # Check exponential growth roughly (allow jitter): second >= first
    assert sleeps[1] >= sleeps[0]

def test_non_retryable_raises(monkeypatch, client):
    client.model = DummyModel([Exception("invalid_argument: bad request")])
    with pytest.raises(Exception):
        client._retry_generate("prompt", max_retries=3)

def test_exhaust_retries(monkeypatch, client):
    client.model = DummyModel([Exception("503")] * 5)
    calls = []
    monkeypatch.setattr("services.vertex_client.time.sleep", lambda s: calls.append(s))
    with pytest.raises(Exception):
        client._retry_generate("prompt", max_retries=2)  # 1st try + 2 retries
    assert len(calls) == 2  # slept exactly for the two retries

def test_should_retry_messages(client):
    assert client._should_retry(Exception("429 Too Many Requests"))
    assert client._should_retry(Exception("quota exceeded"))
    assert client._should_retry(Exception("503 Service Unavailable"))
    assert not client._should_retry(Exception("unauthenticated"))
    assert not client._should_retry(Exception("permission_denied"))
    assert not client._should_retry(Exception("invalid_argument"))