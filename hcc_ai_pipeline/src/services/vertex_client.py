from loguru import logger
import os
import json
import re
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel


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

    def extract_conditions(self, text: str) -> list[str]:
        prompt = f"""Extract medical conditions from the assessment/plan section below.

Return ONLY a valid JSON array of condition names. Do not include any explanations, markdown, or other text.
Example format: ["Condition 1", "Condition 2", "Condition 3"]

Assessment/Plan section:
{text}
"""

        try:
            response = self.model.generate_content(prompt)
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
        response_text = response.text.strip() if response.text else ""
        
        # Try multiple parsing strategies
        parsed_conditions = None
        
        # Strategy 1: Try parsing the entire response as JSON
        try:
            parsed_conditions = json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Try extracting JSON from markdown code blocks
        if parsed_conditions is None:
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed_conditions = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Try finding the first JSON array in the text
        if parsed_conditions is None:
            # Find the first [ and try to parse from there
            start_idx = response_text.find('[')
            if start_idx != -1:
                # Try to find the matching closing bracket
                bracket_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '[':
                        bracket_count += 1
                    elif response_text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            try:
                                parsed_conditions = json.loads(response_text[start_idx:i+1])
                            except json.JSONDecodeError:
                                pass
                            break
        
        # Validate and return the parsed conditions
        if parsed_conditions is not None:
            if isinstance(parsed_conditions, list):
                # Ensure all items are strings and filter out empty values
                conditions = [str(cond).strip() for cond in parsed_conditions if cond]
                if conditions:
                    return conditions
                else:
                    logger.warning("LLM returned empty list of conditions")
                    return []
            else:
                logger.warning(f"LLM returned non-list type: {type(parsed_conditions)}; returning empty list")
                return []
        else:
            logger.warning("Failed to parse LLM output as JSON")
            logger.debug(f"LLM response text: {response_text[:500]}")  # Log first 500 chars for debugging
            return []
