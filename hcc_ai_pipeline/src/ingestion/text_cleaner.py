def extract_assessment_section(text: str) -> str:
    lower = text.lower()
    marker = "assessment/plan"

    if marker not in lower:
        return text

    idx = lower.index(marker)
    return text[idx + len(marker):].strip()
