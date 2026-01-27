import re

# Start headers: robust match for Assessment/Plan variants (case-insensitive, multiline)
# - Accepts: "Assessment / Plan", "Assessment/Plan", "Assessment and Plan", "Assessment & Plan", "A/P"
# - Tolerates extra spaces and optional trailing ":" or dash-like punctuation
_START_HEADER_RE = re.compile(
    r"""(?imx)                    # i: ignore case, m: ^/$ anchor per line, x: verbose regex
    ^\s*(                         # line start, allow leading spaces
        a\s*/\s*p                 # "A/P"
      | assessment\s*             # "assessment"
        (?:/|\s*(?:and|&)\s*)\s*  # either "/" or "and"/"&" with flexible spacing
        plan                      # "plan"
      | assessment\s+plan         # "assessment plan" (space only)
    )\s*[:\-–—]?\s*$              # optional colon/dash, then end of line
    """
)

# End headers: likely delimiters for where the section ends in your samples
# - Seen in pn_1..pn_9: "Return to Office", "Encounter Sign-Off"
# - Also include common variants that often follow the A/P
_END_HEADER_RE = re.compile(
    r"""(?imx)
    ^\s*(
        return\s+to\s+office
      | encounter\s+sign[-\s]?off
      | follow[\s\-]?up
      | disposition
      | medications? | meds
      | plan\s*:?
    )\s*$
    """
)

def extract_assessment_section(text: str) -> str:
    """
    Extract the text under an 'Assessment/Plan' style header up to the next header.

    Behavior:
    - If a start header is found, return text from immediately after that header
      until the next recognized end header (or end-of-file if none).
    - If no start header is found, return the original text (safe fallback).
    - If the extracted section is empty after trimming, return the original text
      to avoid losing content.
    """
    if not text:
        return text

    # Locate the first start header line that matches any supported variant
    start = _START_HEADER_RE.search(text)
    if not start:
        # No clear A/P header: preserve the full note rather than risk truncation
        return text

    start_idx = start.end()

    # Find the next header after the start to delimit the end of the section.
    # If not found, we assume A/P runs to the end of the document.
    end = _END_HEADER_RE.search(text, pos=start_idx)
    end_idx = end.start() if end else len(text)

    # Extract and trim the section between start and end
    section = text[start_idx:end_idx].strip()

    # If extraction yields nothing (e.g., malformed document), fall back safely
    return section if section else text