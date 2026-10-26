"""
Deterministic guardrails that check answers against tool results.
"""

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 4000
KM2_PATTERN = re.compile(r"(\d[\d,']*(?:\.\d+)?)\s*(?:km²|km2|km\^2|sq\.?\s*km|square\s+kilomet(?:er|re)s?)", re.IGNORECASE)
RELATIVE_TOLERANCE = 0.01
ABSOLUTE_TOLERANCE = 0.01


@dataclass(frozen=True)
class GuardrailReport:
    ungrounded_values: list[float]

    @property
    def ok(self) -> bool:
        return not self.ungrounded_values


# Returns an error message for unusable input, otherwise None
def validate_user_input(text: str) -> str | None:
    if not text or not text.strip():
        return "Please enter a question."
    
    if len(text) > MAX_INPUT_CHARS:
        return f"The question is too long ({len(text)} characters, maximum {MAX_INPUT_CHARS})."
    
    return None


# Accept the rounding the model may apply when presenting a value (e.g. 12.3456 written as 12.3 or 12)
def _matches(value: float, reference: float) -> bool:
    for digits in (4, 3, 2, 1, 0):
        if abs(round(reference, digits) - value) <= ABSOLUTE_TOLERANCE:
            return True
    return abs(value - reference) <= RELATIVE_TOLERANCE * max(abs(reference), 1e-9)


def extract_km2_values(text: str) -> list[float]:
    values = []
    for match in KM2_PATTERN.finditer(text):
        try:
            values.append(float(match.group(1).replace(",", "").replace("'", "")))
        except ValueError:
            continue
    return values


# Every km^2 number in the answer must correspond to a value returned by calculate_affected_area (or the AOI size from it)
def check_area_grounding(answer: str, tool_area_values: list[float]) -> GuardrailReport:
    ungrounded = [value for value in extract_km2_values(answer) if not any(_matches(value, reference) for reference in tool_area_values)]
    return GuardrailReport(ungrounded_values=ungrounded)


UNGROUNDED_WARNING = ("Automatic check: the answer mentions area values ({values} km²) that do not match any result of calculate_affected_area in this conversation. "
                      "Treat them as unverified. Only areas from the deterministic area tool are reliable.")
