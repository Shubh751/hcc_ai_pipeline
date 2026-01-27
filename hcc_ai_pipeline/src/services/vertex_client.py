from loguru import logger
import os
import json
import re
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel
from app.state import Condition

# Added for retry/backoff support and optional Google error typing
import time
import random
try:
    from google.api_core import exceptions as gexc
except Exception:
    gexc = None


class VertexLLMClient:
    def __init__(self, project: str, location: str):
        logger.info("Initializing Vertex AI client")
        
        # Validate credentials before initializing
        self._validate_credentials()
        
        try:
            vertexai.init(project=project, location=location)
            self.model = GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            error_msg = str(e)
            if "EndOfStreamError" in error_msg or "pyasn1" in error_msg:
                raise ValueError(
                    "Invalid or corrupted Google Cloud credentials file. "
                    "Please check that GOOGLE_APPLICATION_CREDENTIALS points to a valid service account JSON file. "
                    "The credentials file may be corrupted, incomplete, or in an invalid format."
                ) from e
            raise
    
    def _validate_credentials(self):
        """Validate that credentials file exists and is readable."""
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not creds_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. "
                "Please set it to the path of your Google Cloud service account JSON file."
            )
        
        creds_file = Path(creds_path)
        if not creds_file.exists():
            raise FileNotFoundError(
                f"Credentials file not found: {creds_path}. "
                "Please check that GOOGLE_APPLICATION_CREDENTIALS points to an existing file."
            )
        
        # Try to parse the JSON to validate format
        try:
            with open(creds_file, 'r') as f:
                creds_data = json.load(f)
                # Check for required fields
                if "private_key" not in creds_data or "client_email" not in creds_data:
                    raise ValueError(
                        f"Invalid credentials file format: {creds_path}. "
                        "The file must be a valid service account JSON with 'private_key' and 'client_email' fields."
                    )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Credentials file is not valid JSON: {creds_path}. "
                f"Error: {e}"
            ) from e

    def extract_conditions(self, text: str) -> list[Condition]:
        prompt = f"""Extract medical conditions from the assessment/plan section below.
                    Return ONLY a valid JSON array of objects with a 'name' field. Do not include any explanations, markdown, or other text.
                    Example format:
                    [ {{"name": "Hypertension"}}, {{"name": "Type 2 diabetes mellitus"}} ]
                    Assessment/Plan section:
                    {text}
                    """

        try:
            # Use retry with exponential backoff and jitter for transient failures
            response = self._retry_generate(prompt)
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            # Check for credential-related errors
            if "EndOfStreamError" in error_type or "pyasn1" in error_msg or "EndOfStreamError" in error_msg:
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "not set")
                raise ValueError(
                    f"Invalid or corrupted Google Cloud credentials file: {creds_path}\n"
                    "The private key in your service account JSON file cannot be parsed.\n"
                    "This usually means:\n"
                    "  1. The credentials file is corrupted or incomplete\n"
                    "  2. The private_key field is truncated or malformed\n"
                    "  3. The file was edited incorrectly\n\n"
                    "Please download a fresh service account key from Google Cloud Console:\n"
                    "  https://console.cloud.google.com/iam-admin/serviceaccounts\n"
                    "Then set GOOGLE_APPLICATION_CREDENTIALS to the path of the new file."
                ) from e

            # Re-raise other errors as-is
            raise

        # Parse the response text
        response_text = response.text.strip() if getattr(response, "text", None) else ""

        # Strategy 1: parse full text as JSON
        parsed = None
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = None

        # Strategy 2: extract JSON from fenced code blocks
        if parsed is None:
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    parsed = None

        # Strategy 3: find the first JSON array substring
        if parsed is None:
            start_idx = response_text.find('[')
            if start_idx != -1:
                bracket_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '[':
                        bracket_count += 1
                    elif response_text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            try:
                                parsed = json.loads(response_text[start_idx:i+1])
                            except json.JSONDecodeError:
                                parsed = None
                            break

        # Validate and normalize to List[Condition]
        if parsed is None:
            logger.warning("Failed to parse LLM output as JSON")
            logger.debug(f"LLM response text: {response_text[:500]}")  # first 500 chars for debugging
            return []

        if not isinstance(parsed, list):
            logger.warning(f"LLM returned non-list type: {type(parsed)}; returning empty list")
            return []

        conditions: list[Condition] = []

        # Preferred: list of dicts with 'name'
        if parsed and isinstance(parsed[0], dict):
            for obj in parsed:
                name = str(obj.get("name", "")).strip()
                if not name:
                    continue
                code_val = obj.get("code")
                code = None if code_val in (None, "", "nan") else str(code_val).strip()
                hcc_flag = bool(obj.get("hcc_relevant")) if "hcc_relevant" in obj else False
                conditions.append(Condition(name=name, code=code, hcc_relevant=hcc_flag))

        # Back-compat: list of strings
        elif parsed and isinstance(parsed[0], str):
            for s in parsed:
                s_norm = str(s).strip()
                if s_norm:
                    conditions.append(Condition(name=s_norm))

        if not conditions:
            logger.warning("LLM returned no valid conditions after normalization")
            return []

        return conditions


    def _should_retry(self, exc: Exception) -> bool:
        # Prefer Google API typed exceptions when available; otherwise fallback to message heuristics.
        if gexc and isinstance(exc, (gexc.ResourceExhausted, gexc.DeadlineExceeded, gexc.ServiceUnavailable, gexc.InternalServerError, gexc.Unknown)):
            return True
        msg = str(exc).lower()
        # Treat quota/timeouts/unavailability as retryable.
        if any(s in msg for s in ["resource_exhausted", "quota", "temporar", "deadlineexceeded", "unavailable", "internal", "502", "503", "504", "429"]):
            return True
        # Treat auth/permission/invalid-argument and 4xx as non-retryable.
        if any(s in msg for s in ["unauthenticated", "permission_denied", "invalid_argument", "credentials", "forbidden", "400", "401", "403"]):
            return False
        # Default to non-retryable to avoid retry storms on unknown errors.
        return False

    def _retry_generate(self, prompt: str, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 8.0):
        # Exponential backoff with full jitter per attempt:
        # delay = min(max_delay, base_delay * 2^attempt) * (0.5 + random())
        attempt = 0
        while True:
            try:
                return self.model.generate_content(prompt)
            except Exception as e:
                # Stop if non-retryable or attempts exhausted; propagate original error.
                if not self._should_retry(e) or attempt >= max_retries:
                    raise
                sleep_s = min(max_delay, base_delay * (2 ** attempt)) * (0.5 + random.random())
                time.sleep(sleep_s)
                attempt += 1
                logger.error(f"retrying {attempt} times")