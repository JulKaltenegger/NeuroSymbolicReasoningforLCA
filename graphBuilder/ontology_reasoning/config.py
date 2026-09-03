"""Profiles, env loading, LLM settings."""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
GRAPH_BUILDER_DIR = _PACKAGE_DIR.parent
REPO_ROOT = GRAPH_BUILDER_DIR.parent
OWL_DIR = REPO_ROOT / "owl"
DEFAULT_ONTOLOGY_FILENAME = "KB-LCA-merged.ttl"
DEFAULT_ONTOLOGY_PATH = OWL_DIR / DEFAULT_ONTOLOGY_FILENAME
_ENV_LOADED = False



# . config.py runs first, at import time. 
# Its _load_env() executes at module level, reading .env, and it defines the three profiles (bbsr, tabula, slice) 
# that decide everything downstream: which branches are retrieved, whether the LLM runs all tasks, including 
# layer_function tasks, or just material tasks, in full or material_only mode. 
# The similarity threshold, and top_k. Nothing else can run before this


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    for path in (REPO_ROOT / ".env", GRAPH_BUILDER_DIR / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()

PROFILES = {
    "bbsr": {
        "branches": ("material", "layer_function", "layerset"),
        "llm_mode": "full",
        "use_llm": True,
        "top_k": 10,
        "retrieval_top_k": 4,
        "similarity_threshold": 0.25,
    },
    "tabula": {
        "branches": ("material", "layer_function", "layerset"),
        "llm_mode": "full",
        "use_llm": True,
        "top_k": 8,
        "retrieval_top_k": 4,
        "similarity_threshold": 0.25,
    },
    "slice": {
        "branches": ("material", "layer_function"),
        "llm_mode": "material_only",
        "use_llm": True,
        "top_k": 8,
        "retrieval_top_k": 6,
        "similarity_threshold": 0.30,
    },
}

DEFAULT_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()
DEFAULT_MODELS = {
    "ollama": os.environ.get("OLLAMA_MODEL", "llama3"),
    "openai": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "google": os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash"),
}
# [UNUSED] callers read get_llm_config()["model"] instead
# DEFAULT_LLM_MODEL = DEFAULT_MODELS["ollama"]
# [UNUSED] stale re-export — retrieval.py imports LAYER_FUNCTION_URI from layer_axioms
# LAYER_FUNCTION_URI = "https://w3id.org/bmp#LayerFunction"
DCTERMS_DESCRIPTION = "http://purl.org/dc/terms/description"


def resolve_llm_provider() -> str:
    if LLM_PROVIDER in ("ollama", "openai", "google"):
        return LLM_PROVIDER
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "google"
    return "ollama"


def get_llm_config() -> dict:
    provider = resolve_llm_provider()
    return {"provider": provider, "model": DEFAULT_MODELS.get(provider, DEFAULT_MODELS["ollama"])}


def get_profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}. Choose from: {list(PROFILES)}")
    return PROFILES[name]


def resolve_ontology_path(owl_dir: Path | None = None) -> Path:
    """Canonical KB path: owl/KB-LCA-merged.ttl (override with ONTOLOGY_PATH)."""
    explicit = os.environ.get("ONTOLOGY_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            path = REPO_ROOT / explicit
        if not path.is_file():
            raise FileNotFoundError(explicit)
        return path.resolve()
    owl_dir = owl_dir or OWL_DIR
    path = owl_dir / DEFAULT_ONTOLOGY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def resolve_ontology_overlay_path(owl_dir: Path | None = None) -> Path | None:
    """Optional overlay TTL parsed after the main KB (ONTOLOGY_OVERLAY_PATH only)."""
    explicit = os.environ.get("ONTOLOGY_OVERLAY_PATH", "").strip()
    if not explicit:
        return None
    path = Path(explicit)
    if not path.is_file():
        path = (owl_dir or OWL_DIR) / explicit
    if not path.is_file():
        path = REPO_ROOT / explicit
    if not path.is_file():
        raise FileNotFoundError(explicit)
    return path.resolve()
