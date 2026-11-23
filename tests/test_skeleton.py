"""Skeleton check: every scamguard module imports and documents its purpose. Offline, no API."""

import importlib

import pytest

MODULES = [
    "scamguard",
    "scamguard.config",
    "scamguard.schemas",
    "scamguard.llm",
    "scamguard.ocr",
    "scamguard.sanitizer",
    "scamguard.extractor",
    "scamguard.verifier",
    "scamguard.detector",
    "scamguard.risk",
    "scamguard.gate",
    "scamguard.advisor",
    "scamguard.pipeline",
    "scamguard.cli",
    "scamguard.api",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_with_docstring(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip()
