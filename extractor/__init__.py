"""
extractor package
==================
OCR-free receipt data extractors for SmartSplit Bill AI.

Supported models
----------------
GeminiExtractor  — Google Gemini 3 Flash (API, no local GPU needed)
DonutExtractor   — Donut CORD v2 (local, ~800 MB, GPU recommended)
"""

from .gemini_extractor import GeminiExtractor
from .donut_extractor  import DonutExtractor

__all__ = ["GeminiExtractor", "DonutExtractor"]
