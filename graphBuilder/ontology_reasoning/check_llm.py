"""Verify .env / API keys without printing full secrets.

    python graphBuilder/run_pipeline.py --check-llm
    python graphBuilder/run_pipeline.py --check-llm --live
    python -m ontology_reasoning.check_llm [--live]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import GRAPH_BUILDER_DIR, REPO_ROOT, get_llm_config
from .llm_backends import chat_json, is_llm_available


def mask_secret(value: str | None) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _scan_env_file_issues(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.is_file():
        return issues
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and any(
            token in stripped
            for token in ("OPENAI_API_KEY=", "GOOGLE_API_KEY=", "GEMINI_API_KEY=", "LLM_PROVIDER=")
        ):
            name = stripped.lstrip("# ").split("=", 1)[0]
            issues.append(f"Commented out (remove leading #): {name}")
        if stripped.startswith("Get key:"):
            issues.append(f"Should be a comment (add # at start): {stripped[:60]}...")
    return issues


def check_setup() -> int:
    print("=== LLM / API key check ===\n")

    for path in (REPO_ROOT / ".env", GRAPH_BUILDER_DIR / ".env"):
        status = "found" if path.is_file() else "missing"
        print(f"  {path}  [{status}]")

    loaded = [p for p in (REPO_ROOT / ".env", GRAPH_BUILDER_DIR / ".env") if p.is_file()]
    if loaded:
        print(f"\nLoaded: {', '.join(str(p) for p in loaded)}")
    else:
        print("\nNo .env file loaded. Copy .env.example to .env and add your key.")

    for path in loaded:
        issues = _scan_env_file_issues(path)
        if issues:
            print("\n.env format issues:")
            for issue in issues:
                print(f"  ! {issue}")

    cfg = get_llm_config()
    available = is_llm_available()
    print(f"\nResolved provider: {cfg['provider']}")
    print(f"Resolved model:    {cfg['model']}")
    print(f"LLM available:     {available}")

    print("\nKeys (masked):")
    print(f"  OPENAI_API_KEY:  {mask_secret(os.environ.get('OPENAI_API_KEY'))}")
    print(f"  GOOGLE_API_KEY:  {mask_secret(os.environ.get('GOOGLE_API_KEY'))}")
    print(f"  GEMINI_API_KEY:  {mask_secret(os.environ.get('GEMINI_API_KEY'))}")

    if not available:
        print("\nNOT READY: set an API key in .env or use local Ollama.")
        return 1

    print("\nREADY: python graphBuilder/run_pipeline.py")
    return 0


def ping_llm() -> int:
    print("\n=== Live LLM ping (configured provider) ===")
    try:
        result = chat_json(
            'Return JSON only: {"ok": true}',
            "Reply with ok true.",
        )
    except Exception as exc:
        print(f"  Status: FAILED\n  {exc}")
        return 1
    print("  Status: OK")
    print(f"  Response: {result}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    live = "--live" in argv
    code = check_setup()
    if live:
        if code:
            print("\nSkipping live ping (setup is not ready).")
            return code
        code |= ping_llm()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
