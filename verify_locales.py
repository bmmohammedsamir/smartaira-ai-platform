"""Verify translation schema parity and placeholder parity for all locale files."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


LOCALES_DIR = Path(__file__).resolve().parent / "locales"
LANGUAGES = ("ar", "en", "fr", "it", "de")
PLACEHOLDER_RE = re.compile(r"\{([^{}!:]+)(?:[!:][^{}]+)?\}")


def flatten(value: object, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten(nested_value, nested_prefix))
        return flattened
    if not isinstance(value, str):
        raise TypeError(f"{prefix} must be a string")
    return {prefix: value}


def load_language(language: str) -> dict[str, str]:
    with (LOCALES_DIR / f"{language}.json").open(encoding="utf-8") as locale_file:
        return flatten(json.load(locale_file))


def main() -> int:
    locales = {language: load_language(language) for language in LANGUAGES}
    reference_keys = set(locales["ar"])
    errors: list[str] = []

    for language, entries in locales.items():
        keys = set(entries)
        missing = sorted(reference_keys - keys)
        extra = sorted(keys - reference_keys)
        if missing:
            errors.append(f"{language}: missing keys: {', '.join(missing)}")
        if extra:
            errors.append(f"{language}: extra keys: {', '.join(extra)}")
        for key, value in entries.items():
            if not value.strip():
                errors.append(f"{language}: empty value for {key}")
            if key in locales["ar"]:
                expected = set(PLACEHOLDER_RE.findall(locales["ar"][key]))
                actual = set(PLACEHOLDER_RE.findall(value))
                if actual != expected:
                    errors.append(
                        f"{language}: placeholder mismatch for {key}; expected {sorted(expected)}, got {sorted(actual)}"
                    )

    if errors:
        print("Locale verification failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Locale verification passed: {len(reference_keys)} keys across {len(LANGUAGES)} languages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
