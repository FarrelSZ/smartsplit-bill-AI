"""
extractor/donut_extractor.py
==============================
Extracts structured bill data from a receipt image using
Donut — a Document Understanding Transformer that is OCR-free.

The model is fine-tuned on the CORD dataset (Clova OCR Receipt Dataset),
a large-scale Korean receipt corpus, so it understands common receipt
layouts without needing an explicit OCR step.

Model   : naver-clova-ix/donut-base-finetuned-cord-v2
Paper   : Kim et al. (2022) — "OCR-Free Document Understanding Transformer"
          https://arxiv.org/abs/2111.15664
HF Hub  : https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2
Weights : ~800 MB (downloaded automatically on first use)

Note on Indonesian receipts
---------------------------
CORD v2 is trained on Korean receipts. Field accuracy on Indonesian
receipts may be lower, especially for less-standard layouts.
This is worth noting in the qualitative comparison with Gemini.
"""

import re
from functools import cached_property

from PIL import Image

_MODEL_ID    = "naver-clova-ix/donut-base-finetuned-cord-v2"
_TASK_PROMPT = "<s_cord-v2>"


class DonutExtractor:
    """
    OCR-free receipt extractor using Donut fine-tuned on CORD v2.

    The model and processor are lazy-loaded on the first call to
    .extract() to avoid slow imports when the class is instantiated.
    GPU (CUDA) is used automatically when available; falls back to CPU.

    Usage
    -----
    >>> extractor = DonutExtractor()
    >>> bill_data = extractor.extract(pil_image)   # first call downloads weights
    """

    @cached_property
    def _device(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def _processor(self):
        from transformers import DonutProcessor
        return DonutProcessor.from_pretrained(_MODEL_ID)

    @cached_property
    def _model(self):
        from transformers import VisionEncoderDecoderModel
        import torch
        m = VisionEncoderDecoderModel.from_pretrained(_MODEL_ID)
        m.eval()
        return m.to(self._device)

    # ── Public API ────────────────────────────────────────────────────────────
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
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for the Donut model.\n"
                "Install with: pip install torch transformers sentencepiece\n"
                "Or switch to the Gemini model (no local install needed)."
            ) from exc

        processor = self._processor
        model     = self._model
        device    = self._device

        # Prepare the task token prompt
        decoder_input_ids = processor.tokenizer(
            _TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        # Encode image pixels
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Generate output sequence
        with torch.no_grad():
            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode tokens → CORD JSON-like string
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "")
        seq = seq.replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()   # remove task token

        cord_data = processor.token2json(seq)
        return self._parse_cord(cord_data)

    # ── Private helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _parse_cord(cord: dict) -> dict:
        """
        Convert CORD v2 output format → standard bill dict.

        CORD schema reference
        ----------------------
        {
          "menu": [{"nm": "", "cnt": "", "price": "", "unitprice": ""}],
          "sub_total": {
              "subtotal_price": "",
              "service_price": "",
              "tax_price": "",
              "discount_price": ""
          },
          "total": {"total_price": "", "cashprice": "", "changeprice": ""}
        }
        """

        def clean_num(s) -> float:
            """Strip non-numeric chars and return float."""
            if not s:
                return 0.0
            try:
                return float(re.sub(r"[^\d.]", "", str(s)) or "0")
            except ValueError:
                return 0.0

        # ── Items ──
        menu = cord.get("menu", [])
        if isinstance(menu, dict):   # single item returned as dict, not list
            menu = [menu]

        items = []
        for entry in menu:
            name      = str(entry.get("nm", "Unknown Item"))
            qty_raw   = entry.get("cnt",       "1")
            unit_raw  = entry.get("unitprice", "0")
            total_raw = entry.get("price",     "0")

            try:
                qty = max(1, int(re.sub(r"[^\d]", "", str(qty_raw)) or "1"))
            except ValueError:
                qty = 1

            unit_price  = clean_num(unit_raw)
            total_price = clean_num(total_raw)

            # Derive missing value from the other
            if total_price == 0 and unit_price > 0:
                total_price = unit_price * qty
            if unit_price == 0 and total_price > 0 and qty > 0:
                unit_price = total_price / qty

            items.append(
                {
                    "name":        name,
                    "qty":         qty,
                    "unit_price":  unit_price,
                    "total_price": total_price,
                }
            )

        # ── Sub-total & additional charges ──
        sub = cord.get("sub_total", {})
        subtotal = clean_num(sub.get("subtotal_price", 0))

        additional = []
        if sub.get("service_price"):
            additional.append({"name": "Service Charge", "amount":  clean_num(sub["service_price"])})
        if sub.get("tax_price"):
            additional.append({"name": "Tax",            "amount":  clean_num(sub["tax_price"])})
        if sub.get("discount_price"):
            additional.append({"name": "Discount",       "amount": -clean_num(sub["discount_price"])})

        # ── Total ──
        total_data = cord.get("total", {})
        total      = clean_num(total_data.get("total_price", 0))

        # Fallback calculations when fields are missing
        if subtotal == 0 and items:
            subtotal = sum(i["total_price"] for i in items)
        if total == 0:
            total = subtotal + sum(c["amount"] for c in additional)

        return {
            "items":              items,
            "subtotal":           subtotal,
            "additional_charges": additional,
            "total":              total,
        }
