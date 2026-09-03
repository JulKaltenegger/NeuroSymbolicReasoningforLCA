"""Source-specific description records for the OWL NLP pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DescriptionRecord:
    subject_uri: str
    german_desc: str = ""
    english_desc: str = ""
    extra_text: str | None = None
    use_llm: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
