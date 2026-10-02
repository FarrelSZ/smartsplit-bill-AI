"""
SmartSplit Bill AI — Main Application
======================================
Streamlit-based AI receipt scanner & bill splitter.
Supports Gemini 3 Flash (API) and Donut CORD v2 (local).

Author : [Your Name]
Course : Data Science & ML Bootcamp — Dibimbing
"""

import json
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSplit Bill AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Global CSS & Design Tokens
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background-color: #F0F4FF; }

/* Hide default Streamlit chrome (keep header so the sidebar toggle stays accessible) */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent;
}
header[data-testid="stHeader"]::before { display: none; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    border-radius: 20px;
    padding: 38px 28px;
    text-align: center;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(79,70,229,0.25);
}
.hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0 0 6px; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; opacity: 0.88; margin: 0; }

/* ── Step label & title ── */
.step-label {
    font-size: 0.72rem; font-weight: 700; color: #6366F1;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px;
}
.step-title {
    font-size: 1.6rem; font-weight: 800; color: #1E1B4B; margin: 0 0 24px;
}

/* ── Progress bar ── */
.prog-bar { display: flex; gap: 6px; margin-bottom: 28px; align-items: center; }
.prog-seg { flex: 1; height: 7px; border-radius: 99px; background: #D4D8FF; }
.prog-seg.done   { background: #4F46E5; }
.prog-seg.active { background: linear-gradient(90deg, #4F46E5, #7C3AED); }

/* ── Generic card ── */
.card {
    background: white; border-radius: 14px; padding: 22px;
    border: 1px solid #E0E7FF;
    box-shadow: 0 2px 8px rgba(79,70,229,0.07);
    margin-bottom: 14px;
}

/* ── Person summary card ── */
.person-card {
    background: white; border-radius: 14px; padding: 20px;
    border: 2px solid #E0E7FF;
    box-shadow: 0 2px 8px rgba(79,70,229,0.06);
}
.person-name  { font-size: 1.05rem; font-weight: 700; color: #1E1B4B; margin-bottom: 10px; }
.item-line    { font-size: 0.875rem; color: #4B5563; padding: 4px 0; border-bottom: 1px dashed #F0F0F0; }
.person-total { font-size: 1.3rem; font-weight: 800; color: #4F46E5; margin-top: 12px; }

/* ── Status banners ── */
.banner-ok   { background:#D1FAE5; border:1.5px solid #10B981; border-radius:10px; padding:12px 16px; color:#065F46; font-weight:600; }
.banner-warn { background:#FEF3C7; border:1.5px solid #F59E0B; border-radius:10px; padding:12px 16px; color:#92400E; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #C7D2FE !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3  { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.12) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
}

/* ── Sidebar collapse / expand toggle — keep it clearly visible ── */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] button * {
    color: #FFFFFF !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}
/* Expand chevron in the header when the sidebar is collapsed */
[data-testid="stBaseButton-headerNoPadding"] {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    margin-left: 8px !important;
}
[data-testid="stBaseButton-headerNoPadding"]:hover {
    background: #7C3AED !important;
}
[data-testid="stBaseButton-headerNoPadding"] * {
    color: #FFFFFF !important;
}

/* ── Primary button ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; color: white !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
STEPS = ["Upload", "Review", "People", "Assign", "Summary"]


# ──────────────────────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────────────────────
def _init() -> None:
    defaults = {
        "step":           0,          # 0-indexed, matches STEPS list
        "image":          None,
        "bill_data":      None,
        "inference_time": 0.0,
        "model_used":     "",
        "people":         [],
        "assignments":    {},         # {str(item_idx): person_name}
        "model_choice":   "gemini",   # default model
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
def _sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧾 SmartSplit Bill AI")
        st.caption("Scan · Split · Settle 💸")
        st.markdown("---")

        st.markdown("### ⚙️ Model Config")
        st.selectbox(
            "AI Model",
            options=["gemini", "donut"],
            format_func=lambda x: (
                "🌟 Gemini 3 Flash (API)"
                if x == "gemini"
                else "🍩 Donut CORD v2 (Local)"
            ),
            key="model_choice",
            help="Gemini → requires API key + internet. Donut → runs locally, no key needed.",
        )

        if st.session_state.model_choice == "gemini":
            # API key is loaded from the .env file (GEMINI_API_KEY)
            if os.getenv("GEMINI_API_KEY"):
                st.success("✅ API key loaded from .env")
            else:
                st.error("❌ GEMINI_API_KEY not found in .env file.")
        else:
            st.info("⚙️ First run downloads ~800 MB model weights from Hugging Face.")

        st.markdown("---")
        st.markdown("### 📍 Progress")
        cur = st.session_state.step
        icons = ["📸", "📊", "👥", "🎯", "💰"]
        for i, label in enumerate(STEPS):
            mark = "✅" if i < cur else ("**🔵**" if i == cur else "⬜")
            st.markdown(f"{mark} {icons[i]} {label}")

        st.markdown("---")
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────
def _progress_html(current: int) -> str:
    segs = []
    for i in range(len(STEPS)):
        cls = "done" if i < current else ("active" if i == current else "")
        segs.append(f'<div class="prog-seg {cls}"></div>')
    return f'<div class="prog-bar">{"".join(segs)}</div>'


def _fmt(val: float) -> str:
    """Format value as Rupiah string, e.g. Rp 15.000"""
    return f"Rp {int(round(val)):,}".replace(",", ".")


def _to_float(val) -> float:
    """Safely convert any value to float (strips non-numeric chars)."""
    import re
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(re.sub(r"[^\d.]", "", str(val)) or "0")
    except ValueError:
        return 0.0


def _calculate_splits(
    bill_data: dict,
    people: list[str],
    assignments: dict,
) -> tuple[dict, dict]:
    """
    Compute per-person payment totals.

    - Each item is fully charged to its assigned person.
    - Additional charges (tax/service) are split equally.
    - Rounding: last person absorbs any floating-point remainder so the
      sum of all portions equals the original total charge exactly.

    Returns
    -------
    totals : dict {person: float}  — amount each person owes
    detail : dict {person: list}   — items assigned to each person
    """
    items      = bill_data.get("items", [])
    additional = bill_data.get("additional_charges", [])

    totals = {p: 0.0 for p in people}
    detail = {p: [] for p in people}

    # 1. Assign item costs
    for idx, item in enumerate(items):
        person = assignments.get(str(idx)) or (people[0] if people else None)
        if person and person in totals:
            price = _to_float(item.get("total_price", 0))
            totals[person] += price
            detail[person].append({"name": item.get("name", "?"), "price": price})

    # 2. Split additional charges equally, fix rounding on last person
    n = len(people)
    if n:
        total_extra = sum(_to_float(c.get("amount", 0)) for c in additional)
        base_share  = round(total_extra / n, 2)
        shares      = [base_share] * n
        diff        = round(total_extra - sum(shares), 2)
        shares[-1] += diff                                  # absorb rounding diff
        for person, share in zip(people, shares):
            totals[person] = round(totals[person] + share, 2)

    return totals, detail


# ──────────────────────────────────────────────────────────────
# STEP 0 — Upload & Extract
# ──────────────────────────────────────────────────────────────
def step_upload() -> None:
    st.markdown(_progress_html(0), unsafe_allow_html=True)
    st.markdown(
        '<p class="step-label">Step 1 of 5</p>'
        '<p class="step-title">Scan Your Receipt 📸</p>',
        unsafe_allow_html=True,
    )

    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        file = st.file_uploader(
            "Upload receipt image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True, caption="Receipt Preview")

            model_label = (
                "Gemini 3 Flash" if st.session_state.model_choice == "gemini" else "Donut CORD v2"
            )
            if st.button(
                f"🚀 Scan with {model_label}",
                type="primary",
                use_container_width=True,
            ):
                if (
                    st.session_state.model_choice == "gemini"
                    and not os.getenv("GEMINI_API_KEY")
                ):
                    st.error("❌ GEMINI_API_KEY not found. Add it to your .env file and restart the app.")
                    return

                with st.spinner("🤖 AI is reading your receipt…"):
                    try:
                        t0 = time.perf_counter()

                        if st.session_state.model_choice == "gemini":
                            from extractor.gemini_extractor import GeminiExtractor
                            bill_data = GeminiExtractor().extract(image)
                        else:
                            from extractor.donut_extractor import DonutExtractor
                            bill_data = _get_donut().extract(image)

                        elapsed = time.perf_counter() - t0

                        st.session_state.image          = image
                        st.session_state.bill_data      = bill_data
                        st.session_state.inference_time = elapsed
                        st.session_state.model_used     = st.session_state.model_choice
                        st.session_state.step           = 1
                        st.rerun()

                    except Exception as exc:
                        st.error(f"❌ Extraction failed: {exc}")

    with col_info:
        st.markdown(
            """
            <div class="card">
                <b>📋 Tips for best results</b>
                <ul style="margin-top:12px; padding-left:20px; color:#374151; line-height:1.8;">
                    <li>Use a well-lit, clear photo</li>
                    <li>All text should be readable, not blurry</li>
                    <li>Keep the receipt flat — avoid crumples</li>
                    <li>Avoid heavy shadows across the text</li>
                </ul>
            </div>
            <div class="card">
                <b>🤖 Supported Models</b>
                <p style="color:#374151; margin-top:10px; font-size:0.88rem; line-height:1.6;">
                    <b>Gemini 3 Flash</b><br>
                    Google multimodal LLM — excellent accuracy for Indonesian receipts.
                    Requires internet & API key.
                </p>
                <hr style="border-color:#E0E7FF; margin:10px 0;">
                <p style="color:#374151; font-size:0.88rem; line-height:1.6;">
                    <b>Donut CORD v2</b><br>
                    OCR-free transformer fine-tuned on receipts. Runs fully offline.
                    Best on standard receipt layouts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# STEP 1 — Review & Edit Extracted Data
# ──────────────────────────────────────────────────────────────
def step_review() -> None:
    st.markdown(_progress_html(1), unsafe_allow_html=True)
    st.markdown(
        '<p class="step-label">Step 2 of 5</p>'
        '<p class="step-title">Review Extracted Data 📊</p>',
        unsafe_allow_html=True,
    )

    bill_data   = st.session_state.bill_data
    model_label = "Gemini 3 Flash" if st.session_state.model_used == "gemini" else "Donut CORD v2"
    t           = st.session_state.inference_time

    st.info(f"⚡ **{model_label}** extracted data in **{t:.2f}s** — review and correct if needed.")

    col_img, col_data = st.columns([1, 2], gap="large")

    # Left: receipt preview
    with col_img:
        if st.session_state.image:
            st.image(st.session_state.image, use_container_width=True, caption="Receipt")

    # Right: editable data
    with col_data:

        # ── Items table (editable) ──
        st.markdown("#### 🛍️ Items")
        items_df = pd.DataFrame(
            bill_data.get("items", []),
            columns=["name", "qty", "unit_price", "total_price"],
        )
        for col in ["qty", "unit_price", "total_price"]:
            items_df[col] = pd.to_numeric(items_df[col], errors="coerce").fillna(0)

        edited_df = st.data_editor(
            items_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "name":        st.column_config.TextColumn("Item Name",  width="large"),
                "qty":         st.column_config.NumberColumn("Qty",       format="%d",    min_value=0),
                "unit_price":  st.column_config.NumberColumn("Unit Price",format="Rp %.0f",min_value=0),
                "total_price": st.column_config.NumberColumn("Subtotal",  format="Rp %.0f",min_value=0),
            },
            key="items_editor",
        )
        bill_data["items"] = edited_df.to_dict("records")

        # ── Pricing summary (editable) ──
        st.markdown("#### 💰 Pricing")
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            sub = st.number_input(
                "Subtotal (Rp)",
                value=float(bill_data.get("subtotal", 0)),
                min_value=0.0,
                format="%.0f",
            )
            bill_data["subtotal"] = sub

        with pc2:
            charges     = bill_data.get("additional_charges", [])
            new_charges = []
            for c in charges:
                amt = st.number_input(
                    f"{c['name']} (Rp)",
                    value=float(c.get("amount", 0)),
                    min_value=0.0,
                    format="%.0f",
                    key=f"chg_{c['name']}",
                )
                new_charges.append({"name": c["name"], "amount": amt})
            bill_data["additional_charges"] = new_charges

            if st.button("➕ Add Charge", use_container_width=True):
                new_charges.append({"name": "New Charge", "amount": 0.0})
                bill_data["additional_charges"] = new_charges
                st.rerun()

        with pc3:
            total = st.number_input(
                "**Total Bill (Rp)**",
                value=float(bill_data.get("total", 0)),
                min_value=0.0,
                format="%.0f",
            )
            bill_data["total"] = total

        # Balance check
        extra_sum = sum(_to_float(c["amount"]) for c in bill_data.get("additional_charges", []))
        calc_total = sub + extra_sum
        if abs(calc_total - total) < 1:
            st.markdown('<div class="banner-ok">✅ Totals balanced.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="banner-warn">⚠️ Items+charges = {_fmt(calc_total)} '
                f'but bill total = {_fmt(total)}. Please adjust.</div>',
                unsafe_allow_html=True,
            )

        st.session_state.bill_data = bill_data

    st.markdown("<br>", unsafe_allow_html=True)
    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 0
            st.rerun()
    with nc2:
        if st.button("Continue →", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()


# ──────────────────────────────────────────────────────────────
# STEP 2 — Add People
# ──────────────────────────────────────────────────────────────
def step_people() -> None:
    st.markdown(_progress_html(2), unsafe_allow_html=True)
    st.markdown(
        '<p class="step-label">Step 3 of 5</p>'
        "<p class=\"step-title\">Who's Splitting? 👥</p>",
        unsafe_allow_html=True,
    )

    col_form, col_list = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Add Participants")
        with st.form("add_person", clear_on_submit=True):
            name = st.text_input("Name", placeholder="e.g. Budi")
            if st.form_submit_button("➕ Add Person", use_container_width=True):
                name = name.strip()
                if not name:
                    st.warning("Name cannot be empty.")
                elif name in st.session_state.people:
                    st.warning(f'"{name}" already added.')
                else:
                    st.session_state.people.append(name)
                    st.rerun()

    with col_list:
        people = st.session_state.people
        st.markdown(f"#### {len(people)} person(s) added")
        if people:
            for person in people:
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"👤 **{person}**")
                with c2:
                    if st.button("✕", key=f"rm_{person}"):
                        st.session_state.people.remove(person)
                        # Remove stale assignments for this person
                        st.session_state.assignments = {
                            k: v
                            for k, v in st.session_state.assignments.items()
                            if v != person
                        }
                        st.rerun()
        else:
            st.info("Add at least one person to continue.")

    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with nc2:
        if st.button(
            "Continue →",
            type="primary",
            use_container_width=True,
            disabled=(len(st.session_state.people) == 0),
        ):
            st.session_state.step = 3
            st.rerun()


# ──────────────────────────────────────────────────────────────
# STEP 3 — Assign Items
# ──────────────────────────────────────────────────────────────
def step_assign() -> None:
    st.markdown(_progress_html(3), unsafe_allow_html=True)
    st.markdown(
        '<p class="step-label">Step 4 of 5</p>'
        '<p class="step-title">Who Pays for What? 🎯</p>',
        unsafe_allow_html=True,
    )

    bill_data  = st.session_state.bill_data
    people     = st.session_state.people
    items      = bill_data.get("items", [])
    additional = bill_data.get("additional_charges", [])

    col_assign, col_preview = st.columns([3, 2], gap="large")

    with col_assign:
        st.markdown("#### Assign Each Item")

        # Quick-split shortcut
        if st.button("⚡ Split Everything Equally", use_container_width=True):
            for i in range(len(items)):
                st.session_state.assignments[str(i)] = people[i % len(people)]
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        for idx, item in enumerate(items):
            ia, ib, ic = st.columns([4, 2, 2])
            with ia:
                st.markdown(f"**{item.get('name', f'Item {idx+1}')}**")
            with ib:
                price = _to_float(item.get("total_price", 0))
                st.markdown(f"_{_fmt(price)}_")
            with ic:
                current = st.session_state.assignments.get(str(idx), people[0])
                chosen  = st.selectbox(
                    "Pays",
                    options=people,
                    index=people.index(current) if current in people else 0,
                    key=f"sel_{idx}",
                    label_visibility="collapsed",
                )
                st.session_state.assignments[str(idx)] = chosen
            st.divider()

        if additional:
            total_extra = sum(_to_float(c["amount"]) for c in additional)
            per_person  = round(total_extra / len(people), 2)
            names_str   = " + ".join(c["name"] for c in additional)
            st.info(
                f"💡 **{names_str}** total {_fmt(total_extra)} → split equally "
                f"= **{_fmt(per_person)} per person**."
            )

    with col_preview:
        st.markdown("#### Live Totals 👀")
        totals, _ = _calculate_splits(bill_data, people, st.session_state.assignments)
        for person, amt in totals.items():
            st.markdown(
                f'<div class="card" style="margin-bottom:8px; padding:14px 18px;">'
                f'<span style="font-weight:600;color:#1E1B4B;">{person}</span>'
                f'<span style="float:right;font-weight:800;color:#4F46E5;">{_fmt(amt)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        grand = sum(totals.values())
        st.markdown(
            f'<div class="card" style="border:2px solid #4F46E5;padding:14px 18px;">'
            f'<b>Grand Total</b>'
            f'<span style="float:right;font-weight:900;color:#4F46E5;font-size:1.1rem;">'
            f"{_fmt(grand)}</span></div>",
            unsafe_allow_html=True,
        )

    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with nc2:
        if st.button("See Summary →", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


# ──────────────────────────────────────────────────────────────
# STEP 4 — Summary
# ──────────────────────────────────────────────────────────────
def step_summary() -> None:
    st.markdown(_progress_html(4), unsafe_allow_html=True)
    st.markdown(
        '<p class="step-label">Step 5 of 5</p>'
        '<p class="step-title">Bill Split Summary 💰</p>',
        unsafe_allow_html=True,
    )

    bill_data   = st.session_state.bill_data
    people      = st.session_state.people
    assignments = st.session_state.assignments
    additional  = bill_data.get("additional_charges", [])
    total_bill  = _to_float(bill_data.get("total", 0))

    totals, detail = _calculate_splits(bill_data, people, assignments)
    grand_total    = sum(totals.values())

    # ── Per-person cards ──
    n_cols = min(len(people), 3)
    cols   = st.columns(n_cols)

    total_extra  = sum(_to_float(c["amount"]) for c in additional)
    share_extra  = round(total_extra / len(people), 2) if people else 0

    for i, person in enumerate(people):
        with cols[i % n_cols]:
            item_lines = "".join(
                f'<div class="item-line">{d["name"]} — {_fmt(d["price"])}</div>'
                for d in detail[person]
            )
            extra_line = (
                f'<div class="item-line" style="color:#6B7280;">'
                f"Shared charges — {_fmt(share_extra)}</div>"
                if additional
                else ""
            )
            card_html = (
                '<div class="person-card">'
                f'<div class="person-name">👤 {person}</div>'
                f"{item_lines}"
                f"{extra_line}"
                f'<div class="person-total">{_fmt(totals[person])}</div>'
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Bill total verification ──
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Subtotal", _fmt(_to_float(bill_data.get("subtotal", 0))))
    with mc2:
        st.metric("Additional Charges", _fmt(total_extra))
    with mc3:
        st.metric("Total Bill", _fmt(total_bill))

    if abs(grand_total - total_bill) < 1.0:
        st.markdown(
            f'<div class="banner-ok">✅ Verified — all portions sum to '
            f"{_fmt(grand_total)}, matching the bill total.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner-warn">⚠️ Portions sum to {_fmt(grand_total)}, '
            f"but bill total is {_fmt(total_bill)}. Please review the item data.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Export ──
    st.markdown("#### 📥 Export Summary")
    export_data = {
        "bill_total": total_bill,
        "model_used": st.session_state.model_used,
        "people":     people,
        "split":      {
            p: {
                "total": totals[p],
                "items": [d["name"] for d in detail[p]],
            }
            for p in people
        },
    }
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇️ Download Summary (JSON)",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name="smartsplit_summary.json",
            mime="application/json",
            use_container_width=True,
        )
    with ec2:
        if st.button("🔄 Split Another Bill", type="primary", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ──────────────────────────────────────────────────────────────
# Model cache — load Donut weights only once per session
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def _get_donut():
    from extractor.donut_extractor import DonutExtractor
    return DonutExtractor()


# ──────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────
_STEP_FNS = [step_upload, step_review, step_people, step_assign, step_summary]


def main() -> None:
    _init()
    _sidebar()

    st.markdown(
        """
        <div class="hero">
            <h1>🧾 SmartSplit Bill AI</h1>
            <p>Scan any receipt · Split it fairly · Settle instantly</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _STEP_FNS[st.session_state.step]()


if __name__ == "__main__":
    main()
