"""
extractor/gemini_extractor.py
==============================
Extracts structured bill data from a receipt image
using Google Gemini 3 Flash (multimodal LLM via API).

This is an OCR-free approach: the raw pixel data is sent
directly to the model, which understands text without an
explicit OCR pre-processing step.

Model  : gemini-3-flash-preview
HF Hub : N/A (Google API)
Docs   : https://ai.google.dev/gemini-api/docs
"""

import json
import os
import re
import time

import google.generativeai as genai
from PIL import Image

# ── Structured extraction prompt ─────────────────────────────────────────────
_PROMPT = """
You are a receipt data extractor. Analyze the receipt/bill image carefully and
return ONLY a single valid JSON object — no preamble, no markdown fences.

Required JSON structure:
{
  "items": [
    {
      "name": "Exact item name from the receipt",
      "qty": 1,
      "unit_price": 15000,
      "total_price": 15000
    }
  ],
  "subtotal": 30000,
  "additional_charges": [
    {"name": "Tax PPN 11%", "amount": 3300},
    {"name": "Service Charge 5%", "amount": 1500}
  ],
  "total": 34800
}

Extraction rules:
- ALL price values must be plain numbers (int or float) — NO currency symbols, NO dots/commas as separators
- Extract EVERY item visible on the receipt
- subtotal = sum of all item total_prices (before any extra charges)
- additional_charges = any tax, service charge, discount, or other fee shown separately
- total = the final bill amount printed at the bottom
- If a field is not visible, use 0 (items) or [] (additional_charges)
- Return ONLY the JSON object.
"""


class GeminiExtractor:
    """
    OCR-free receipt extractor using Gemini 3 Flash.

    The receipt image is sent as a multimodal input together
    with a structured prompt. The model returns a JSON object
    which is then parsed and normalised.

    Usage
    -----
    >>> extractor = GeminiExtractor()
    >>> bill_data = extractor.extract(pil_image)
    """

    def __init__(self, model_name: str = "gemini-3-flash-preview") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set.\n"
                "Set it in the sidebar or add it to your .env file."
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def extract(self, image: Image.Image) -> dict:
        """
        Extract bill data from a receipt image.

        Parameters
        ----------
        image : PIL.Image.Image
            Receipt photo in RGB mode.

        Returns
        -------
        dict
            Keys: items, subtotal, additional_charges, total
        """
        # Free tier allows ~5 requests/minute — retry with backoff on 429
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content([_PROMPT, image])
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)  # 5s, 10s, 20s, 40s, 60s
                    time.sleep(wait)
                else:
                    raise

        raw      = response.text.strip()

        # Strip markdown code fences if the model adds them despite instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",          "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned non-JSON output.\n"
                f"Raw response: {raw[:300]}\n"
                f"JSONDecodeError: {exc}"
            ) from exc

        return self._normalize(data)

    # ── Private helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _normalize(data: dict) -> dict:
        """Ensure all numeric fields are floats and structure is consistent."""

        def to_f(v) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            try:
                cleaned = re.sub(r"[^\d.]", "", str(v))
                return float(cleaned) if cleaned else 0.0
            except ValueError:
                return 0.0

        items = [
            {
                "name":        str(item.get("name", "Unknown Item")),
                "qty":         int(item.get("qty", 1) or 1),
                "unit_price":  to_f(item.get("unit_price",  0)),
                "total_price": to_f(item.get("total_price", 0)),
            }
            for item in data.get("items", [])
        ]

        additional = [
            {
                "name":   str(c.get("name", "Charge")),
                "amount": to_f(c.get("amount", 0)),
            }
            for c in data.get("additional_charges", [])
        ]

        return {
            "items":              items,
            "subtotal":           to_f(data.get("subtotal", 0)),
            "additional_charges": additional,
            "total":              to_f(data.get("total",    0)),
        }
