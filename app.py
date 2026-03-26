"""
Novontis AI — Novontis Beauty Expert  (v3.1)
============================================
Base: v3.0 document code (question_hint style, natural opener).

Added in v3.1 — gender & age_range from photo only:
─────────────────────────────────────────────────────
+ _FACE_ANALYSIS_PROMPT now returns "gender" and "age_range" fields.
+ _SAFE_DEFAULTS and _FACE_ANALYSIS_PROMPT_SIMPLE updated to match.
+ analyze_face_with_gpt() returns "gender" and "age_range" in features dict.
+ build_onboarding_prompt() reads gender/age from features — injects silent
  tone calibration. User is NEVER asked about these.
+ build_system_prompt() reads gender/age from features — scopes product advice
  and adjusts conversational energy by age group.
+ Profile badge redesigned: colour-coded chips show gender/age from face
  analysis alongside skin data collected during onboarding.

All other logic (5-question onboarding, question_hint flow, conversation
chaining, product search, YouTube) is unchanged from the v3.0 document.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"]       = "3"

import base64
import io
import json
import math
import re
import time
import urllib.parse
import uuid
from html import escape as html_escape

import requests
import streamlit as st
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Novontis - Your Beauty Expert",
    page_icon="✨",
    layout="centered",
)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

VISION_MODEL = "gpt-4o"
CHAT_MODEL   = "gpt-4o-mini"

MAX_HISTORY_MESSAGES = 20
ANALYSIS_IMG_MAX_PX  = 1024
CHAT_IMG_MAX_PX      = 512

BUDGET_LABELS: dict[str, str] = {
    "low":  "under ₹500",
    "mid":  "₹500–₹2,000",
    "high": "above ₹2,000",
}

APPROVED_INFLUENCERS = [
    "Hyram", "NikkieTutorials", "Jackie Aina", "Susan Yara",
    "Gothamista", "Dr Dray", "Andrea Matillano", "Hindash",
    "Shreya Jain", "Malvika Sitlani",
]

BRAND_AFFILIATE_OVERRIDES: dict[str, str] = {}

if not OPENAI_API_KEY:
    st.error(
        "**OPENAI_API_KEY is not set.**  "
        "Add it to your `.env` file and restart:\n\n"
        "```\nOPENAI_API_KEY=sk-...\n```"
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
html, body { background: #faf5f0 !important; }

.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section[data-testid="stMain"],
.main .block-container {
    background: #faf5f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div {
    background: #faf5f0 !important;
    border-top: 1px solid #ecd8cc !important;
    box-shadow: 0 -4px 20px rgba(180,100,60,0.07) !important;
}

.stChatInputContainer,
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background: #ffffff !important;
    border: 1.5px solid #e0c8b8 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(180,90,50,0.10) !important;
}
.stChatInputContainer textarea,
[data-testid="stChatInputTextArea"],
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #2b1a1a !important;
}
.stChatInputContainer textarea::placeholder,
[data-testid="stChatInput"] textarea::placeholder { color: #c0a090 !important; }
[data-testid="stChatInputSubmitButton"] button,
.stChatInputContainer button {
    background: linear-gradient(135deg, #c96a40, #e08050) !important;
    border: none !important; border-radius: 8px !important; color: #fff !important;
}

[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border-radius: 18px !important;
    border: 1px solid #edddd4 !important;
    padding: 1rem 1.1rem !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 2px 8px rgba(160,80,40,0.06) !important;
}
[data-testid="stChatMessageContent"] *,
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] strong {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    color: #2b1a1a !important;
}
[data-testid="stChatMessageContent"] strong {
    font-weight: 600 !important; color: #5a2a0a !important;
}

.forence-typing {
    display: flex; align-items: center; gap: 6px; padding: 6px 0;
}
.forence-typing .dot {
    width: 8px; height: 8px;
    background: #c96a40; border-radius: 50%;
    animation: forence-bounce 1.3s infinite ease-in-out;
}
.forence-typing .dot:nth-child(1) { animation-delay: 0s; }
.forence-typing .dot:nth-child(2) { animation-delay: 0.2s; }
.forence-typing .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes forence-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
    40%           { transform: translateY(-7px); opacity: 1; }
}
.forence-typing-label {
    color: #a07060; font-size: 0.80rem; font-style: italic; margin-left: 4px;
}

.f-card {
    border: 1px solid #edddd4; border-radius: 14px;
    padding: 14px 12px; background: #fffaf6;
    text-align: center; display: flex; flex-direction: column;
    align-items: center;
    box-shadow: 0 2px 8px rgba(160,80,40,0.06); height: 100%;
}
.f-card-name {
    font-weight: 700; font-size: 0.80rem; color: #2b1a1a;
    line-height: 1.35; min-height: 52px;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
    width: 100%; margin-bottom: 4px;
}
.f-card-reason {
    font-size: 0.68rem; color: #a07050; font-style: italic;
    min-height: 18px; margin-bottom: 4px;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden; width: 100%;
}
.f-card-brand {
    color: #b08060; font-size: 0.70rem; text-transform: uppercase;
    letter-spacing: 0.05em; min-height: 20px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; width: 100%; margin-bottom: 4px;
}
.f-card-stars {
    color: #e8912a; font-size: 0.80rem; min-height: 22px;
    display: flex; align-items: center; justify-content: center;
    gap: 2px; margin-bottom: 4px;
}
.f-card-rating-count { color: #aaa; font-size: 0.68rem; }
.f-card-price {
    font-weight: 800; font-size: 0.92rem; color: #2b1a1a;
    min-height: 26px; display: flex; align-items: center;
    justify-content: center; margin-bottom: 10px;
}
.f-card-price-na {
    color: #ccc; font-size: 0.75rem; min-height: 26px;
    display: flex; align-items: center;
    justify-content: center; margin-bottom: 10px;
}
.f-shop-btn {
    display: block;
    background: linear-gradient(135deg, #c96a40, #e08050);
    color: #fff !important; text-decoration: none !important;
    padding: 8px 0; border-radius: 20px;
    font-size: 0.76rem; font-weight: 700;
    transition: opacity .15s; width: 100%;
    box-sizing: border-box; margin-top: auto;
}
.f-shop-btn:hover { opacity: .85; }
.f-no-link-btn {
    display: block; padding: 8px 0; border-radius: 20px;
    font-size: 0.72rem; color: #ccc; margin-top: auto; width: 100%;
    text-align: center;
}
.f-disclaimer {
    font-size: 0.66rem; color: #c0a090; font-style: italic;
    margin: 2px 0 8px 0; text-align: left;
}
.f-reasoning {
    font-size: 0.80rem; color: #8a5030; margin: 0 0 10px 0;
    font-style: italic;
}
.f-section-label {
    font-size: 0.92rem; font-weight: 700; color: #a05030;
    margin: 14px 0 4px 0;
}

.yt-card {
    border: 1px solid #edddd4; border-radius: 12px;
    overflow: hidden; background: #fff;
    box-shadow: 0 2px 8px rgba(160,80,40,0.06);
}
.yt-card img { width: 100%; height: 110px; object-fit: cover; }
.yt-card-body { padding: 8px 10px 10px 10px; }
.yt-card-title {
    font-size: 0.76rem; font-weight: 600; color: #2b1a1a;
    line-height: 1.35;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 4px;
}
.yt-card-channel { font-size: 0.68rem; color: #a07060; margin-bottom: 8px; }
.yt-watch-btn {
    display: block; background: #ff0000; color: #fff !important;
    text-decoration: none !important; text-align: center;
    padding: 5px 0; border-radius: 6px;
    font-size: 0.72rem; font-weight: 700;
}
.yt-watch-btn:hover { opacity: .85; }

/* v3.1: colour-coded chip card replaces flat pill badge */
.profile-card {
    background: #ffffff; border: 1px solid #edddd4;
    border-radius: 16px; padding: 10px 14px;
    box-shadow: 0 2px 12px rgba(160,80,40,0.07); margin-top: 4px;
}
.profile-card-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.profile-chip {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.75rem; font-weight: 500; white-space: nowrap;
    border: 1px solid transparent;
}
.chip-face    { background: #fdf4ee; border-color: #e0c4b0; color: #5a2a0a; }
.chip-skin    { background: #fef9f5; border-color: #ddc8b8; color: #5a2a0a; }
.chip-tone    { background: #fff5f0; border-color: #e8c8b8; color: #5a2a0a; }
.chip-feature { background: #fdfaf7; border-color: #ddd0c8; color: #7a4030; }
.chip-gender  { background: #f5f0fe; border-color: #c8b8e8; color: #40307a; }
.chip-age     { background: #f0f5fe; border-color: #b8c8e8; color: #30507a; }
.chip-stype   { background: #f0faf4; border-color: #a8d8b8; color: #1a6030; }
.chip-concern { background: #fffbf0; border-color: #e8d898; color: #7a6010; }
.chip-allergy { background: #fff0f0; border-color: #e8b8b8; color: #7a2020; }
.chip-budget  { background: #f0fbf0; border-color: #a8e8a8; color: #206020; }
.chip-pref    { background: #f8f0fe; border-color: #d0b8e8; color: #5a2080; }

h1 {
    font-family: 'DM Serif Display', serif !important;
    color: #2b1a1a !important; font-size: 2rem !important;
    letter-spacing: -0.3px;
}
.subtitle { font-size: 0.92rem; color: #9a7060; margin-top: 4px; margin-bottom: 1.5rem; }
.step-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: #c07050; margin-bottom: 0.75rem;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c96a40, #e08050) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 500 !important;
    box-shadow: 0 4px 14px rgba(180,90,50,0.28) !important;
}
div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #a05030 !important;
    border: 1.5px solid #e0c4b0 !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important;
}

[data-testid="stFileUploader"] {
    background: #fff !important; border: 1.5px dashed #e0c4b0 !important;
    border-radius: 16px !important; padding: 1rem !important;
}
[data-testid="stImage"] img {
    border-radius: 14px !important; border: 2.5px solid #f0d8c8 !important;
    box-shadow: 0 3px 14px rgba(160,80,40,0.14) !important;
}
hr { border-color: #edddd4 !important; }
.stSpinner > div { color: #c47f5a !important; }
</style>
""",
    unsafe_allow_html=True,
)


def _pil_to_base64(pil_image: Image.Image, max_px: int = 1024, quality: int = 85) -> str:
    img = pil_image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── v3.1 ADDITION: gender and age_range added to face analysis prompt ─────────
_FACE_ANALYSIS_PROMPT = (
    "You are a professional beauty analyst specialising in face morphology and skin science.\n\n"
    "Look at this photo carefully. A face may be at any angle, in any lighting, or partially "
    "obscured — do your best analysis with what is visible. Never refuse to analyse.\n\n"
    "Return ONLY a valid JSON object with exactly this structure. "
    "No preamble, no markdown fences, no extra text:\n\n"
    "{\n"
    '  "face_shape": "<oval|round|square|heart|diamond|oblong>",\n'
    '  "skin_tone": "<fair|light|medium|tan|deep>",\n'
    '  "undertone": "<warm|cool|neutral>",\n'
    '  "texture_notes": "<concise: e.g. smooth, slight dryness, visible pores on T-zone>",\n'
    '  "notable_features": "<beauty-relevant observations: e.g. high cheekbones, almond eyes, strong jawline>",\n'
    '  "brightness_estimate": <integer 0-255, where 240=very fair, 180=medium, 100=tan, 60=deep>,\n'
    '  "gender": "<male|female|non-binary|unknown>",\n'
    '  "age_range": "<teens|20s|30s|40s|50s|60plus|unknown>"\n'
    "}\n\n"
    "gender: infer from visual cues (hair, features, styling). Use 'unknown' if genuinely unclear.\n"
    "age_range: estimate from skin and facial structure. Use 'unknown' if not enough signal.\n"
    "If any other field is unclear, make your best reasonable estimate — "
    "do NOT return an error. Only set face_shape to null if you genuinely cannot "
    "see any face at all in the image."
)

_FACE_ANALYSIS_PROMPT_SIMPLE = (
    "Look at the person in this photo and give me a quick beauty analysis. "
    "Return ONLY a JSON object, no other text:\n"
    '{"face_shape":"oval","skin_tone":"medium","undertone":"neutral",'
    '"texture_notes":"smooth","notable_features":"","brightness_estimate":150,'
    '"gender":"unknown","age_range":"unknown"}\n\n'
    "Replace those values with your actual observations from the photo. "
    "Use your best guess even if the lighting or angle is not perfect."
)

_REQUIRED_FIELDS = {"face_shape", "skin_tone", "undertone"}

_SAFE_DEFAULTS: dict = {
    "face_shape":          "oval",
    "skin_tone":           "medium",
    "undertone":           "neutral",
    "texture_notes":       "",
    "notable_features":    "",
    "brightness_estimate": 150,
    "gender":              "unknown",
    "age_range":           "unknown",
}


def _call_gpt_face_analysis(b64: str, prompt: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}", "detail": "high"
                    }},
                ],
            }],
            max_tokens=400,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        if not any(k in parsed for k in _REQUIRED_FIELDS):
            return None
        return parsed
    except Exception:
        return None


def analyze_face_with_gpt(pil_image: Image.Image) -> dict:
    b64    = _pil_to_base64(pil_image, max_px=ANALYSIS_IMG_MAX_PX)
    parsed = _call_gpt_face_analysis(b64, _FACE_ANALYSIS_PROMPT)

    if parsed is not None:
        print("Face analysis successful on first attempt.")
        print("PARSED ANALYSIS:", parsed)

    if parsed is None:
        parsed = _call_gpt_face_analysis(b64, _FACE_ANALYSIS_PROMPT_SIMPLE)

    if parsed is None:
        print("Using safe defaults for face analysis.")
        parsed = dict(_SAFE_DEFAULTS)

    for k, v in _SAFE_DEFAULTS.items():
        parsed.setdefault(k, v)

    brightness   = int(parsed.get("brightness_estimate", 150))
    bright_label = "fair" if brightness > 180 else "medium" if brightness > 100 else "deep"

    return {
        "face_shape":  parsed.get("face_shape") or "oval",
        "skin_color": {
            "undertone":  parsed.get("undertone") or "neutral",
            "brightness": brightness,
            "tone_label": bright_label,
        },
        "texture_metrics": {
            "texture_notes": parsed.get("texture_notes", ""),
            "variance":      0.0,
            "edge_density":  0.0,
        },
        "skin_tone":        parsed.get("skin_tone", bright_label),
        "notable_features": parsed.get("notable_features", ""),
        "gender":           parsed.get("gender",    "unknown"),
        "age_range":        parsed.get("age_range", "unknown"),
    }


def _build_image_context_messages(b64_chat: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Here's my photo — keep it as a visual reference "
                    "throughout our conversation for personalised advice."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_chat}", "detail": "low"
                }},
            ],
        },
        {
            "role": "assistant",
            "content": (
                "Got your photo! I'll keep it in mind throughout our chat — "
                "let's find what works best for you. 💫"
            ),
        },
    ]


def _safe_parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m2 = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m2:
        try:
            reply_text = m2.group(1).encode("utf-8").decode("unicode_escape")
        except Exception:
            reply_text = m2.group(1)
        return {"reply": reply_text, "signals": {}}
    return None


_KNOWN_BRANDS = [
    "Neutrogena", "Cetaphil", "CeraVe", "L'Oreal", "Loreal", "Maybelline",
    "NYX", "MAC", "Clinique", "Estee Lauder", "Lancome", "Shiseido",
    "Olay", "Garnier", "Nivea", "Dove", "Ponds", "Lakme", "Lotus",
    "Biotique", "Himalaya", "Mamaearth", "Plum", "mCaffeine",
    "Minimalist", "Re'equil", "Fixderma", "The Derma Co", "Foxtale",
    "Pilgrim", "WOW", "St.Botanica", "Forest Essentials",
    "Kama Ayurveda", "VLCC", "Jovees", "Everyuth", "Vaseline",
    "La Roche-Posay", "Vichy", "Avene", "Bioderma", "COSRX",
    "The Ordinary", "Paula's Choice", "Drunk Elephant", "Tatcha",
    "Sunday Riley", "Glow Recipe", "Innisfree", "Laneige",
    "Revlon", "Colorbar", "Sugar", "Faces Canada", "Nykaa",
    "Schwarzkopf", "Pantene", "TRESemme", "OGX", "Moroccanoil",
    "Charlotte Tilbury", "Urban Decay", "Too Faced", "Benefit",
    "NARS", "Fenty", "Pat McGrath", "Huda Beauty", "Rare Beauty",
    "e.l.f.", "Wet n Wild", "Milani", "ColourPop", "Indulekha",
]


def _extract_brand(title: str) -> str:
    tl = title.lower()
    for brand in _KNOWN_BRANDS:
        if brand.lower() in tl:
            return brand
    return title.split()[0].capitalize() if title else "Unknown"


def _parse_price(raw) -> float:
    if not raw: return 0.0
    try: return round(float(re.sub(r"[^\d.]", "", str(raw))), 2)
    except ValueError: return 0.0


def _parse_rating(raw) -> float:
    if not raw: return 0.0
    m = re.search(r"(\d+\.?\d*)", str(raw))
    return float(m.group(1)) if m else 0.0


def _parse_count(raw) -> int:
    if not raw: return 0
    try: return int(re.sub(r"[^\d]", "", str(raw)))
    except ValueError: return 0


def _is_valid_asin(asin: str) -> bool:
    return bool(asin and re.fullmatch(r"[A-Z0-9]{10}", asin.strip()))


def _make_product_dict(title, image_url, price_raw, rating_raw, count_raw,
                       asin, reason="", category="", skin_type="all",
                       concerns="", distance=0.5) -> dict:
    return {
        "product_id":        str(uuid.uuid4()),
        "name":              title,
        "brand":             _extract_brand(title),
        "category":          category,
        "price":             _parse_price(price_raw),
        "rating":            _parse_rating(rating_raw),
        "rating_count":      _parse_count(count_raw),
        "skin_type":         skin_type,
        "concerns":          concerns,
        "affiliate_amazon":  f"https://www.amazon.in/dp/{asin}",
        "affiliate_cj":      "",
        "affiliate_rakuten": "",
        "image_url":         image_url or "",
        "reason":            reason,
        "_distance":         distance,
        "_asin":             asin,
    }


_GPT_PRODUCT_SYSTEM = (
    "You are a beauty product database assistant for Amazon India.\n\n"
    "STRICT RULES:\n"
    "1. Only include products that are WELL-KNOWN top-sellers on Amazon India "
    "   with thousands of real reviews.\n"
    "2. You MUST include a `confidence` score (0.0-1.0) for each product.\n"
    "   Omit the product entirely if your confidence is below 0.8.\n"
    "3. The ASIN must be exactly 10 uppercase alphanumeric characters "
    "   (e.g. 'B09XJ5K3NP'). If you are not 100% certain of the ASIN, "
    "   OMIT the product - do not guess.\n"
    "4. Return ONLY the JSON array - no markdown, no explanation, no extra keys.\n"
    "5. Limit to {n} products.\n\n"
    "Each element must have exactly these fields:\n"
    "  name, brand, category, price_inr (number), rating (0-5),\n"
    "  review_count (int), asin (10-char), confidence (0-1),\n"
    "  reason (one sentence: why this fits)\n"
)


def _gpt_product_fallback(query: str, filters: dict, n: int = 6) -> list[dict]:
    category  = (filters.get("category")  or "").strip()
    skin_type = (filters.get("skin_type") or "all").strip()
    concerns  = (filters.get("concerns")  or "").strip()
    budget    = (filters.get("budget")    or "").strip()
    avoid     = [a for a in (filters.get("avoid_ingredients") or []) if str(a).lower() != "none"]

    enriched = query
    if category and category not in ("null", ""):         enriched = f"{category} {enriched}"
    if skin_type and skin_type not in ("null", "all", ""): enriched += f" for {skin_type} skin"
    if concerns and concerns not in ("null", ""):          enriched += f" targeting {concerns}"
    if budget in BUDGET_LABELS:                            enriched += f" priced {BUDGET_LABELS[budget]}"
    if avoid:                                              enriched += f" (must NOT contain: {', '.join(avoid[:5])})"

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": _GPT_PRODUCT_SYSTEM.format(n=n)},
                {"role": "user",   "content": (
                    f"Search: {enriched}\n"
                    f"Return up to {n} highly confident real Amazon India products. "
                    f"Drop any product whose ASIN you are not 100% sure of."
                )},
            ],
            temperature=0.2, max_tokens=900,
            response_format={"type": "json_object"},
        )
        data = _safe_parse_json(resp.choices[0].message.content.strip())
        if data is None: return []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list): data = v; break
            else: return []

        products: list[dict] = []
        for i, item in enumerate(data[:n]):
            title      = str(item.get("name",       "")).strip()
            asin       = str(item.get("asin",       "")).strip().upper()
            confidence = float(item.get("confidence", 0) or 0)
            reason     = str(item.get("reason",     "")).strip()
            if not title or not _is_valid_asin(asin) or confidence < 0.8:
                continue
            p = _make_product_dict(
                title=title, image_url="",
                price_raw=item.get("price_inr", 0),
                rating_raw=float(item.get("rating", 0) or 0),
                count_raw=int(item.get("review_count", 0) or 0),
                asin=asin, reason=reason,
                category=str(item.get("category", category)).strip(),
                skin_type=skin_type, concerns=concerns,
                distance=float(i) / max(n, 1),
            )
            p["brand"] = str(item.get("brand", "")).strip() or _extract_brand(title)
            products.append(p)
        return products
    except Exception as e:
        print(f"[GPT product error] {e}")
        return []


def search_products_live(query: str, filters: dict, n: int = 8) -> list[dict]:
    return _gpt_product_fallback(query, filters, n=min(n, 6))


def rank_products(products: list[dict], top_n: int = 4) -> list[dict]:
    for p in products:
        rating    = float(p.get("rating",       0) or 0)
        count     = float(p.get("rating_count", 0) or 0)
        dist      = float(p.get("_distance",    1.0))
        quality   = rating * math.log1p(count)
        relevance = 1.0 / (1.0 + dist)
        p["_score"] = 0.4 * relevance + 0.6 * (quality / 50.0)
    return sorted(products, key=lambda x: x["_score"], reverse=True)[:top_n]


def get_affiliate_link(product: dict) -> tuple[str, str]:
    brand    = product.get("brand", "").lower().strip()
    override = BRAND_AFFILIATE_OVERRIDES.get(brand)
    for net in ([override] if override else []) + ["cj", "rakuten", "amazon"]:
        if product.get(f"affiliate_{net}"):
            return product[f"affiliate_{net}"], net.upper()
    return "", "none"


_MAX_CONCERNS  = 8
_MAX_ALLERGIES = 10


def update_profile(profile: dict, updates: dict) -> dict:
    # gender/age_range are NOT updated here — they come from face analysis only
    if not updates: return profile
    profile = profile.copy()

    if updates.get("skin_type"):
        profile["skin_type"] = updates["skin_type"]

    new_concerns = updates.get("concerns") or []
    if isinstance(new_concerns, str): new_concerns = [new_concerns]
    if new_concerns:
        existing = profile.get("concerns") or []
        if isinstance(existing, str): existing = [existing]
        merged = list(dict.fromkeys(existing + new_concerns))
        profile["concerns"] = merged[:_MAX_CONCERNS]

    new_allergies = updates.get("allergies") or []
    if isinstance(new_allergies, str): new_allergies = [new_allergies]
    if new_allergies:
        existing_a = profile.get("allergies") or []
        if isinstance(existing_a, str): existing_a = [existing_a]
        merged_a = list(dict.fromkeys(existing_a + new_allergies))
        profile["allergies"] = merged_a[:_MAX_ALLERGIES]

    new_budget = (updates.get("budget") or "").strip()
    if new_budget in ("low", "mid", "high"):
        profile["budget"] = new_budget

    if updates.get("preferences"):
        profile["preferences"] = updates["preferences"]

    return profile


def search_youtube_tutorials(query: str, n: int = 3) -> list[dict]:
    influencer_hint = " OR ".join(APPROVED_INFLUENCERS[:4])
    full_query      = f"{query} tutorial ({influencer_hint})"

    if YOUTUBE_API_KEY:
        try:
            params = {
                "part": "snippet", "q": full_query, "type": "video",
                "maxResults": n, "key": YOUTUBE_API_KEY,
                "videoCategoryId": "26", "relevanceLanguage": "en", "safeSearch": "moderate",
            }
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/search", params=params, timeout=8
            )
            if not resp.ok: raise RuntimeError(f"YouTube API HTTP {resp.status_code}")
            tutorials: list[dict] = []
            for item in resp.json().get("items", [])[:n]:
                snippet  = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")
                if not video_id: continue
                tutorials.append({
                    "title":     snippet.get("title", "Tutorial"),
                    "channel":   snippet.get("channelTitle", ""),
                    "url":       f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                    "is_search": False,
                })
            return tutorials
        except Exception as e:
            print(f"[YouTube API error] {e}")

    search_url = (
        "https://www.youtube.com/results?search_query="
        + urllib.parse.quote_plus(full_query)
    )
    return [{"title": f"Search: {query} tutorials", "channel": "YouTube",
             "url": search_url, "thumbnail": "", "is_search": True}]


# ── Onboarding (v3.0 question_hint style — unchanged) ─────────────────────────
_ONBOARDING_QUESTIONS: list[tuple[str, str, str]] = [
    (
        "skin_type",
        (
            "Find out how their skin behaves day to day: does it get oily, "
            "feel dry and tight, or somewhere in between. "
            "Use what you can see in their photo (face shape, skin tone) to make the question feel personal."
        ),
        (
            "Extract the skin type from the reply: dry / oily / combination / "
            "sensitive / normal. Set skin_type in profile_updates."
        ),
    ),
    (
        "concerns",
        (
            "Find out the main thing they want to fix or improve with their skin. "
            "Could be breakouts, dark spots, dullness, uneven texture, anti-ageing, or just building a routine. "
            "Reference what you already know (their skin type) to make the question feel connected."
        ),
        (
            "Extract one or more concerns as a list. "
            "Set concerns in profile_updates."
        ),
    ),
    (
        "allergies",
        (
            "Find out if any ingredients or products have caused reactions or problems for their skin. "
            "Keep it casual, not clinical. Reference their skin type if it helps."
        ),
        (
            "Extract allergens as a list. If user says 'no', 'nothing', or 'none', "
            "set allergies: ['none'] so the question is marked as answered."
        ),
    ),
    (
        "budget",
        (
            "Find out roughly how much they want to spend on beauty products. "
            "Drugstore level, mid-range, or happy to invest more. "
            "Keep it light and conversational, not like filling in a form."
        ),
        (
            "Map to one of: low (under 500 INR) / mid (500-2000 INR) / "
            "high (above 2000 INR). Set budget in profile_updates."
        ),
    ),
    (
        "preferences",
        (
            "Find out if they have any product, brand, or ingredient preferences. "
            "Could be a favourite brand, ingredients they love or want to avoid, "
            "a product type they swear by, or values like vegan or cruelty-free. "
            "Or no preference at all, which is totally fine."
        ),
        (
            "Extract as a string covering any product, brand, ingredient, or value preferences mentioned. "
            "If user says 'none', 'no preference', 'nothing', or 'anything goes', "
            "set preferences: 'none' so the question is marked as answered."
        ),
    ),
]


def _get_onboarding_step(profile: dict) -> int:
    if not profile.get("skin_type"):   return 0
    if not profile.get("concerns"):    return 1
    if not profile.get("allergies"):   return 2
    if not profile.get("budget"):      return 3
    if profile.get("preferences") is None: return 4
    return 5


def build_onboarding_prompt(step: int, profile: dict, features: dict | None,
                            is_opener: bool = False) -> str:
    # v3.1 ADDITION: gender/age from face analysis → silent tone calibration
    gender    = (features or {}).get("gender",    "unknown")
    age_range = (features or {}).get("age_range", "unknown")

    gender_tone = {
        "male":       "User appears male. Use grooming/skincare language. Avoid feminine product terminology.",
        "female":     "User appears female. Full beauty range applies.",
        "non-binary": "User appears non-binary. Use gender-neutral language throughout.",
    }.get(gender, "Gender unclear. Use neutral inclusive language.")

    age_tone = {
        "teens":  "Appears to be a teenager. Light, accessible tone. Focus on basics.",
        "20s":    "In their 20s. Casual, peer-like tone. Basics and trends both work.",
        "30s":    "In their 30s. Warm, direct. Effective routines preferred.",
        "40s":    "In their 40s. Knowledgeable tone. Anti-ageing may be relevant.",
        "50s":    "In their 50s. Thoughtful tone. Skin health and efficacy first.",
        "60plus": "60+. Gentle, patient tone. Comfort and nourishment over trends.",
    }.get(age_range, "Age unclear. Warm inclusive tone for any adult.")

    # Face analysis summary (v3.0 unchanged)
    face_lines: list[str] = []
    if features:
        shape     = features.get("face_shape", "unknown")
        tone      = features["skin_color"].get("tone_label",  "medium")
        undertone = features["skin_color"].get("undertone",   "neutral")
        texture   = features["texture_metrics"].get("texture_notes", "")
        notable   = features.get("notable_features", "")
        face_lines.append(f"Face shape: {shape}, skin tone: {tone} with {undertone} undertone")
        if texture: face_lines.append(f"Texture: {texture}")
        if notable: face_lines.append(f"Notable features: {notable}")
    face_ctx = ("\n".join(f"  {l}" for l in face_lines)) if face_lines else "  Not available"

    collected: list[str] = []
    if profile.get("skin_type"):
        collected.append(f"Skin type  : {profile['skin_type']}")
    if profile.get("concerns"):
        c = profile["concerns"]
        collected.append("Concerns   : " + (", ".join(c) if isinstance(c, list) else c))
    allergies = profile.get("allergies")
    if allergies:
        display_a = [a for a in allergies if a.lower() != "none"]
        collected.append("Allergies  : " + (", ".join(display_a) if display_a else "none reported"))
    if profile.get("budget"):
        collected.append(f"Budget     : {profile['budget']}")
    prefs = profile.get("preferences")
    if prefs:
        collected.append(f"Preferences: {prefs if prefs != 'none' else 'no special preferences'}")
    collected_ctx = ("\n".join(f"  {c}" for c in collected)) if collected else "  Nothing yet"

    field_name, question_hint, extraction_note = _ONBOARDING_QUESTIONS[step]
    has_next = (step + 1) < len(_ONBOARDING_QUESTIONS)

    if has_next:
        next_field_name, next_question_hint, _ = _ONBOARDING_QUESTIONS[step + 1]
        next_q_block = (
            f"NEXT QUESTION (step {step + 2} of 5 - ask this immediately after acknowledging):\n"
            f"  Field   : {next_field_name.replace('_', ' ')}\n"
            f"  Question: \"{next_question_hint}\"\n\n"
            f"CRITICAL FLOW RULE:\n"
            f"When the user clearly answers the current question, your reply MUST do both things:\n"
            f"  1. Acknowledge their answer in ONE warm, natural sentence.\n"
            f"  2. In the SAME message, immediately ask the next question above.\n"
            f"Do NOT send the acknowledgment and the next question as separate messages.\n"
            f"Do NOT say 'Now let me ask...' or 'Moving on...' - just flow naturally into it.\n"
            f"Example flow: \"Got it, [acknowledgment of their answer]. [Next question]?\"\n"
        )
    else:
        next_q_block = (
            "FINAL STEP RULE:\n"
            "When the user answers this last question, your reply MUST do ALL of these in ONE message:\n"
            "  1. Acknowledge their answer in one short warm sentence.\n"
            "  2. Give specific skincare or makeup advice based on everything you know: skin type, "
            "     concerns, budget, face shape, and features from the photo.\n"
            "     Write the advice as a numbered list with EACH STEP ON ITS OWN LINE.\n"
            "     Use Markdown formatting: put a blank line between each numbered step.\n"
            "     Keep each step brief, 1-2 sentences max.\n"
            "  3. Add a short closing line saying product options are below.\n"
            "  4. Set show_products: true with a specific product_query.\n"
            "  5. Set onboarding_field_collected: true.\n\n"
            "FORMATTING — this is critical. The reply field must use actual newlines between steps.\n"
            "WRONG (all on one line): 'Got it! 1. Use a cleanser. 2. Use moisturiser. 3. Use SPF.'\n"
            "RIGHT (each step on its own line):\n"
            "'Got it, no preferences. Here is what I would suggest for your oily skin:\\n\\n"
            "1. Start with a gentle foaming cleanser morning and night to keep oil in check.\\n\\n"
            "2. Use a light oil-free moisturiser, do not skip this even with oily skin.\\n\\n"
            "3. Gel sunscreen every morning, it will feel light and not clog pores.\\n\\n"
            "4. A niacinamide serum a few times a week will help even out your skin tone.\\n\\n"
            "I have pulled some product options below based on your budget.'\n\n"
            "Use \\n\\n between each numbered step so they render on separate lines.\n"
        )

    opener_block = ""
    if is_opener:
        opener_block = (
            "OPENER INSTRUCTION (this is the very first message the user sees):\n"
            "The user has just uploaded their photo. Greet them warmly, mention ONE or TWO specific "
            "things you genuinely notice about their photo (face shape, skin tone, a feature - be "
            "natural and encouraging, never clinical or negative). Do NOT say 'a lot to work with', "
            "'we need to fix', or anything that implies problems. Then naturally ask the first question below.\n"
            "Keep it to 2-3 short sentences total. Sound like a friend, not a form.\n\n"
        )

    return f"""You are Novontis, a warm and knowledgeable beauty expert having a friendly conversation.

FACE ANALYSIS (from the uploaded photo - you can also see the photo directly):
{face_ctx}

TONE GUIDANCE (use silently - never mention or ask about these):
  {gender_tone}
  {age_tone}

WHAT YOU ALREADY KNOW:
{collected_ctx}

{opener_block}CURRENT TASK - step {step + 1} of 5:
Collect the user's {field_name.replace("_", " ")}.

WHAT TO FIND OUT (do NOT copy this — write the question in your own words):
  {question_hint}

IMPORTANT: Every user is different. Write a fresh, natural question based on:
- What you already know about them (face shape, skin tone, anything already collected)
- The conversation so far
- The tone that fits them (age, gender context above)
Do NOT repeat any phrasing you have already used. Do NOT ask it like a form field.

Extraction rule: {extraction_note}

{next_q_block}
TONE AND STYLE RULES:
- Sound like a real person texting a friend. Warm, curious, genuine.
- Use simple everyday words. Never use complex or formal vocabulary. Talk like a normal person, not a professional.
- NEVER use: "Based on your input", "As an AI", "I understand", "Certainly", "Great question", "Absolutely", "stunning", "lovely", "highlight", "beautiful" (when describing user's features).
- STRICTLY NO EM DASHES. The character — is completely banned. If you want to add extra info, use a comma or bracket instead.
  WRONG: "Your skin is oily — that is common for your age."
  RIGHT: "Your skin is oily, which is pretty common."
  WRONG: "I noticed your skin tone — it is warm."
  RIGHT: "I noticed your skin tone is warm."
  Check your reply before outputting. If you see the — character anywhere, rewrite that sentence without it.
- Write flowing sentences. No bullet points, no numbered lists — EXCEPT in the final step where you give routine advice (numbered steps are required there and each step must be on its own line).
- Reference face analysis naturally when relevant, not as a data readout.
- Never imply the user's skin or face has problems that need fixing. Stay positive and encouraging.
- If user goes off topic, acknowledge briefly and bring them back: "Ha, I can look into that later. Right now though, [question]."
- If user asks a beauty question mid-onboarding, give a quick helpful note and return to the question.
- Do NOT recommend products or show tutorials during steps 1-4. ONLY set show_products true on the FINAL step (step 5) when the user answers the last question.
- Always include profile_updates with extracted values when the user answers.
- Set onboarding_field_collected: true when the current question is clearly answered.

RESPONSE FORMAT - return ONLY this JSON, nothing outside it:
{{
  "reply": "Your warm, natural reply. If field collected, include acknowledgment AND next question in same message.",
  "signals": {{
    "onboarding_field_collected": true_or_false,
    "show_products": true_or_false,
    "show_tutorials": false,
    "is_off_topic": false,
    "needs_clarification": false,
    "product_query": "relevant search query based on skin type and concern, or null",
    "product_filters": {{}},
    "product_reasoning": "",
    "tutorial_query": null,
    "profile_updates": {{
      "skin_type":   null,
      "concerns":    [],
      "allergies":   [],
      "budget":      null,
      "preferences": null
    }}
  }}
}}
"""


def generate_opener(features: dict | None, face_b64_chat: str | None) -> str:
    system = build_onboarding_prompt(step=0, profile={}, features=features, is_opener=True)
    messages: list[dict] = [{"role": "system", "content": system}]
    if face_b64_chat:
        messages.extend(_build_image_context_messages(face_b64_chat))
    messages.append({"role": "user", "content": "Hi, I just uploaded my photo."})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages,
            temperature=0.88, max_tokens=300,
            response_format={"type": "json_object"},
        )
        parsed = _safe_parse_json(resp.choices[0].message.content.strip())
        if parsed and parsed.get("reply"):
            reply = parsed["reply"].replace("—", ",").replace("–", ",")
            return reply
    except Exception as e:
        print(f"[opener error] {e}")

    return (
        "Good to see you! I had a look at your photo and I already have some "
        "ideas for you. Before I dive in, I want to ask a few quick things so "
        "my advice actually fits your skin. Does your skin usually feel dry and "
        "tight after washing, or does it tend to get oily through the day?"
    )


def build_system_prompt(profile: dict, features: dict | None) -> str:
    # v3.1 ADDITION: gender/age from features scopes product advice and tone
    gender    = (features or {}).get("gender",    "unknown")
    age_range = (features or {}).get("age_range", "unknown")

    gender_advice = {
        "male":       "User is male. Focus on grooming and skincare. Avoid defaulting to makeup or feminine product language unless they ask.",
        "female":     "User is female. Full beauty range applies - skincare, makeup, haircare, body care, fragrance.",
        "non-binary": "User is non-binary. Use gender-neutral language. Do not assume which product categories interest them.",
    }.get(gender, "Gender unclear. Use neutral language and follow their lead on product categories.")

    age_advice = {
        "teens":  "Teenager. Simple affordable routines. Sun protection basics. No over-complicated multi-step advice.",
        "20s":    "In their 20s. Preventive care and trending products. Casual, curious tone.",
        "30s":    "In their 30s. Effective routines that fit a busy life. Direct advice.",
        "40s":    "In their 40s. Anti-ageing, firmness, and skin health are likely priorities. Knowledgeable tone.",
        "50s":    "In their 50s. Skin health, hydration, and efficacy over trends. Thoughtful, mature tone.",
        "60plus": "60+. Nourishment, comfort, and gentle care. Patient, warm tone.",
    }.get(age_range, "Age unclear. Warm, inclusive tone suitable for any adult.")

    face_ctx = ""
    if features:
        shape         = features.get("face_shape",  "unknown")
        undertone     = features["skin_color"].get("undertone",  "neutral")
        bright_label  = features["skin_color"].get("tone_label", "medium")
        texture_notes = features["texture_metrics"].get("texture_notes", "")
        notable       = features.get("notable_features", "")
        face_ctx = (
            "\n\n━━━ FACE ANALYSIS (from uploaded photo) ━━━\n"
            f"  Face shape        : {shape}\n"
            f"  Skin tone         : {bright_label}, {undertone} undertone\n"
            f"  Gender (detected) : {gender}\n"
            f"  Age range (est.)  : {age_range}\n"
        )
        if texture_notes: face_ctx += f"  Texture           : {texture_notes}\n"
        if notable:       face_ctx += f"  Notable features  : {notable}\n"
        face_ctx += (
            "The user's photo is visually attached - you can see it directly.\n"
            "Reference these details naturally in conversation - NEVER announce "
            "that you are reading analysis data."
        )

    profile_ctx = ""
    parts: list[str] = []
    if profile.get("skin_type"):  parts.append(f"Skin type   : {profile['skin_type']}")
    if profile.get("concerns"):
        c = profile["concerns"]
        parts.append("Concerns    : " + (", ".join(c) if isinstance(c, list) else c))
    if profile.get("allergies"):
        a = profile["allergies"]
        display_a = [x for x in (a if isinstance(a, list) else [a]) if x.lower() != "none"]
        if display_a: parts.append("Allergies   : " + ", ".join(display_a))
    if profile.get("budget") and profile["budget"] in BUDGET_LABELS:
        parts.append(f"Budget      : {BUDGET_LABELS[profile['budget']]}")
    if profile.get("preferences") and str(profile.get("preferences","")).lower() != "none":
        parts.append(f"Preferences : {profile['preferences']}")
    if parts:
        profile_ctx = (
            "\n\n━━━ USER PROFILE (accumulated - use silently, never ask again) ━━━\n"
            + "\n".join(f"  {p}" for p in parts)
        )

    return f"""You are Novontis. You are a real beauty expert who talks like a knowledgeable friend, warm, direct, and genuinely useful. You are not a chatbot or customer service agent.
{face_ctx}{profile_ctx}

PERSON CONTEXT (use silently - never mention or ask about these):
  {gender_advice}
  {age_advice}

YOUR TONE:
- Sound like a real person texting a close friend. Casual, warm, occasionally playful.
- Use simple everyday words. No complex vocabulary. Talk the way a knowledgeable friend would, not a professional writing a report.
- NEVER say: "Based on your input", "As an AI", "I understand your concern", "Certainly", "Of course", "Great question", "I would be happy to help", "stunning", "lovely", "beautiful" (when describing features).
- STRICTLY NO EM DASHES. The character — is completely banned from your replies.
  If you want to add extra context to a sentence, use a comma or rewrite it.
  WRONG: "This serum is lightweight — perfect for oily skin."
  RIGHT: "This serum is lightweight, which makes it great for oily skin."
  WRONG: "Your skin tone is warm — that means gold works well."
  RIGHT: "Your skin tone is warm, so gold shades will work really well."
  Before outputting, check your reply for the — character. If you find one, rewrite that part without it.
- Write in flowing natural sentences. No bullet points, no numbered lists in replies.
- Be confident. Give clear, specific advice. Do not hedge.
- You can see the user's face photo. Reference specific things you notice when it adds genuine value.
- Do not repeat information you have already used in this conversation.

EXPERTISE:
You know everything about skincare, makeup, haircare, fragrance, body care, tools and routines. Give advice specific to this person, not generic advice.

PRODUCT RECOMMENDATIONS:
Always say WHY a product suits this specific person. Mention their skin type, concern, or face shape as the reason. Never just list names.

BUDGET AND ALLERGY RULES:
- Budget known: stay within range. Signal it naturally: "this one is great value" or "a bit of an investment but really worth it for your skin type".
- Allergies known: steer clear automatically. Mention it: "keeping this fragrance-free for you" or "nothing with sulphates in this one".

HANDLING OFF-TOPIC:
If someone asks something unrelated to beauty, acknowledge it briefly and redirect: "Ha, that is a bit outside my area. But speaking of your skin..." Then get back to the conversation.

MANDATORY JSON FORMAT - return ONLY this object, no text outside it:

{{
  "reply": "Your natural, warm, flowing reply. Simple words. No bullet points. The — character is banned completely. Bold (**word**) is fine for product names.",
  "signals": {{
    "onboarding_field_collected": false,
    "show_products": true_or_false,
    "product_query": "specific product search string or null",
    "product_filters": {{
      "skin_type":         "dry|oily|combination|sensitive|all|null",
      "concerns":          "acne|hydration|anti-aging|oil-control|sun-protection|dark-spots|null",
      "category":          "moisturizer|serum|cleanser|sunscreen|foundation|lipstick|toner|eye-cream|null",
      "budget":            "low|mid|high|null",
      "avoid_ingredients": ["ingredient1", "ingredient2"]
    }},
    "product_reasoning": "One sentence: why these specific products suit this user",
    "show_tutorials": true_or_false,
    "tutorial_query": "tutorial search string or null",
    "needs_clarification": true_or_false,
    "is_off_topic": false,
    "profile_updates": {{
      "skin_type":   null,
      "concerns":    [],
      "allergies":   [],
      "budget":      null,
      "preferences": null
    }}
  }}
}}

WHEN TO SET show_products true:
User asks for a recommendation, what to buy, what to try, a dupe, an alternative, or context clearly calls for it.
Never set true while asking a clarification question.

WHEN TO SET show_tutorials true:
User asks "how to apply", "show me how", "tutorial", or after a recommendation where technique matters. Max 3.

PRODUCT FILTER RULES:
Only set a filter if confident. Use null if uncertain.
Always fill avoid_ingredients from known allergies (exclude any "none" entries).
Always fill budget from the known profile.
"""


_FALLBACK_SIGNALS: dict = {
    "onboarding_field_collected": False,
    "show_products":              False,
    "product_query":              None,
    "product_filters":            {},
    "product_reasoning":          "",
    "show_tutorials":             False,
    "tutorial_query":             None,
    "needs_clarification":        False,
    "is_off_topic":               False,
    "profile_updates":            {},
}


def get_forence_response(
    user_input:      str,
    history:         list[dict],
    profile:         dict,
    features:        dict | None,
    face_b64_chat:   str | None = None,
    onboarding_step: int = 5,
) -> dict:
    if onboarding_step < 5:
        system_content = build_onboarding_prompt(onboarding_step, profile, features)
    else:
        system_content = build_system_prompt(profile, features)

    messages: list[dict] = [{"role": "system", "content": system_content}]

    if face_b64_chat:
        messages.extend(_build_image_context_messages(face_b64_chat))

    trimmed = history[:-1]
    if len(trimmed) > MAX_HISTORY_MESSAGES:
        trimmed = trimmed[-MAX_HISTORY_MESSAGES:]
    for msg in trimmed:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    last_error = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL, messages=messages,
                temperature=0.78, max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content.strip()
            parsed = _safe_parse_json(raw)
            if parsed is None:
                last_error = "JSON parse failed"
                time.sleep(0.5)
                continue
            parsed.setdefault("reply", "I'm here! What would you like help with? 💄")
            parsed.setdefault("signals", {})
            for k, v in _FALLBACK_SIGNALS.items():
                parsed["signals"].setdefault(k, v)
            # Hard sanitize: replace any em dashes that slipped through
            if parsed.get("reply"):
                parsed["reply"] = parsed["reply"].replace("—", ",").replace("–", ",")
            return parsed
        except Exception as e:
            last_error = str(e)
            time.sleep(0.5)

    # All 3 attempts failed — show error only now
    return {"reply": "Something went wrong on my end. Give it another go!", "signals": _FALLBACK_SIGNALS}


def _star_html(rating: float) -> str:
    filled = round(max(0.0, min(5.0, rating)))
    return "★" * filled + "☆" * (5 - filled)


def render_product_cards(products: list[dict], reasoning: str = ""):
    if not products: return
    st.markdown("<div class='f-section-label'>🛍️ Products I think you'll love</div>", unsafe_allow_html=True)
    if reasoning:
        st.markdown(f"<p class='f-reasoning'>💡 {html_escape(reasoning)}</p>", unsafe_allow_html=True)

    cols = st.columns(min(len(products), 4))
    for col, product in zip(cols, products):
        rating       = float(product.get("rating",       0) or 0)
        rating_count = int(  product.get("rating_count", 0) or 0)
        price        = float(product.get("price",        0) or 0)
        name_safe    = html_escape(str(product.get("name",   "Unknown Product")))
        brand_safe   = html_escape(str(product.get("brand",  "")))
        reason_safe  = html_escape(str(product.get("reason", "")))

        rating_html = (
            f"<div class='f-card-stars'>{_star_html(rating)}&nbsp;"
            f"<span style='font-size:.76rem;color:#666'>{rating:.1f}</span>"
            f"<span class='f-card-rating-count'>&nbsp;({rating_count:,})</span></div>"
        ) if rating > 0 else (
            "<div class='f-card-stars' style='color:#ddd'>☆☆☆☆☆ "
            "<span style='font-size:.7rem;color:#ccc'>not yet rated</span></div>"
        )
        price_html  = f"<div class='f-card-price'>₹{price:,.0f}</div>" if price > 0 else "<div class='f-card-price-na'>Price unavailable</div>"
        reason_html = f"<div class='f-card-reason'>{reason_safe}</div>" if reason_safe else ""

        with col:
            st.markdown(
                f"""
                <div class="f-card">
                    <div class="f-card-name">{name_safe}</div>
                    {reason_html}
                    <div class="f-card-brand">{brand_safe}</div>
                    {rating_html}
                    {price_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_tutorial_cards(tutorials: list[dict]):
    if not tutorials: return
    st.markdown("<div class='f-section-label'>🎬 Watch & Learn</div>", unsafe_allow_html=True)
    cols = st.columns(min(len(tutorials), 3))
    for col, tut in zip(cols, tutorials):
        title_safe   = html_escape(str(tut.get("title",   "Tutorial")))
        channel_safe = html_escape(str(tut.get("channel", "")))
        url_safe     = html_escape(str(tut.get("url",     "#")))
        thumb_safe   = html_escape(str(tut.get("thumbnail", "")))
        is_search    = tut.get("is_search", False)
        btn_label    = "🔍 Search" if is_search else "▶ Watch"
        thumb_html   = f'<img src="{thumb_safe}" alt="tutorial thumbnail"/>' if thumb_safe else ""
        with col:
            st.markdown(
                f'<div class="yt-card">'
                f'{thumb_html}'
                f'<div class="yt-card-body">'
                f'<div class="yt-card-title">{title_safe}</div>'
                f'<div class="yt-card-channel">{channel_safe}</div>'
                f'<a href="{url_safe}" target="_blank" rel="noopener noreferrer" '
                f'class="yt-watch-btn">{btn_label}</a>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


_DEFAULTS: dict = {
    "features":         None,
    "messages":         [],
    "analyzed":         False,
    "face_image_bytes": None,
    "face_b64_chat":    None,
    "profile": {
        "skin_type":   None,
        "concerns":    [],
        "allergies":   [],
        "budget":      None,
        "preferences": None,
        # gender/age_range are NOT here — they live in features
    },
    "product_results":  {},
    "tutorial_results": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


st.markdown("<h1>✨ Novontis - Your Beauty Expert</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Personalised skincare & makeup advice, powered by AI face analysis.</p>',
    unsafe_allow_html=True,
)
st.divider()


if not st.session_state.analyzed:

    st.markdown('<p class="step-label">Step 1 of 2 · Upload your photo</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a clear, well-lit face photo",
        type=["jpg", "jpeg", "png"],
        help="Front-facing photos with good lighting give the most accurate results.",
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        thumb = pil_image.copy()
        thumb.thumbnail((300, 220), Image.LANCZOS)
        _, col_c, _ = st.columns([1, 2, 1])
        with col_c:
            st.image(thumb, use_column_width=True, caption="Your uploaded photo")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Analyse My Face & Start Chatting →", use_container_width=True, type="primary"):
            with st.spinner("Analysing your features with AI, hang tight..."):
                try:
                    features = analyze_face_with_gpt(pil_image)
                    st.session_state.features = features

                    st.session_state.face_b64_chat = _pil_to_base64(pil_image, max_px=CHAT_IMG_MAX_PX)

                    avatar = pil_image.copy()
                    avatar.thumbnail((120, 120), Image.LANCZOS)
                    buf = io.BytesIO()
                    avatar.save(buf, format="PNG")
                    st.session_state.face_image_bytes = buf.getvalue()

                    st.session_state.analyzed = True

                    opener = generate_opener(
                        features=features,
                        face_b64_chat=st.session_state.face_b64_chat,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": opener})
                    st.rerun()

                except Exception as e:
                    err_str = str(e).lower()
                    if "no clear face" in err_str or "no face" in err_str:
                        st.error(
                            "We could not find a clear face in that photo. "
                            "Try a front-facing shot with good lighting and try again."
                        )
                    else:
                        st.error(
                            "Something went wrong during analysis. "
                            "Please try again, or use a different photo."
                        )

else:

    st.markdown('<p class="step-label">Step 2 of 2 · Your beauty expert</p>', unsafe_allow_html=True)

    f          = st.session_state.features
    shape      = f["face_shape"].capitalize()
    undertone  = f["skin_color"]["undertone"].capitalize()
    tone_label = f["skin_color"]["tone_label"].capitalize()
    p          = st.session_state.profile

    img_col, badge_col = st.columns([1, 4], gap="small")
    with img_col:
        if st.session_state.face_image_bytes:
            st.image(
                io.BytesIO(st.session_state.face_image_bytes),
                use_column_width=True,
                output_format="PNG",
            )

    # v3.1 ADDITION: colour-coded chip card
    with badge_col:
        chips: list[str] = []

        chips.append(f'<span class="profile-chip chip-face"> {html_escape(shape)} face</span>')
        chips.append(f'<span class="profile-chip chip-skin"> {html_escape(tone_label)} skin</span>')
        chips.append(f'<span class="profile-chip chip-tone"> {html_escape(undertone)} undertone</span>')

        notable = f.get("notable_features", "")
        if notable:
            trunc = html_escape(notable[:40]) + ("…" if len(notable) > 40 else "")
            chips.append(f'<span class="profile-chip chip-feature"> {trunc}</span>')

        gender_det = f.get("gender", "unknown")
        if gender_det and gender_det not in ("unknown", ""):
            gender_emoji = {"male": "♂", "female": "♀", "non-binary": "⚧"}.get(gender_det, "👤")
            chips.append(f'<span class="profile-chip chip-gender"> {html_escape(gender_det)}</span>')

        age_det = f.get("age_range", "unknown")
        if age_det and age_det not in ("unknown", ""):
            chips.append(f'<span class="profile-chip chip-age"> {html_escape(age_det)}</span>')

        if p.get("skin_type"):
            chips.append(f'<span class="profile-chip chip-stype"> {html_escape(str(p["skin_type"]))} skin</span>')

        if p.get("concerns"):
            c = p["concerns"]
            c_str = ", ".join(c) if isinstance(c, list) else str(c)
            chips.append(f'<span class="profile-chip chip-concern"> {html_escape(c_str)}</span>')

        if p.get("allergies"):
            display_a = [
                a for a in (p["allergies"] if isinstance(p["allergies"], list) else [p["allergies"]])
                if str(a).lower() != "none"
            ]
            if display_a:
                chips.append(f'<span class="profile-chip chip-allergy"> {html_escape(", ".join(display_a))}</span>')

        if p.get("budget") and p["budget"] in BUDGET_LABELS:
            chips.append(f'<span class="profile-chip chip-budget"> {html_escape(BUDGET_LABELS[p["budget"]])}</span>')

        if p.get("preferences") and str(p.get("preferences", "")).lower() != "none":
            chips.append(f'<span class="profile-chip chip-pref"> {html_escape(str(p["preferences"]))}</span>')

        st.markdown(
            '<div class="profile-card"><div class="profile-card-row">'
            + "".join(chips)
            + "</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and idx in st.session_state.product_results:
                slot = st.session_state.product_results[idx]
                render_product_cards(slot.get("products", []), reasoning=slot.get("reasoning", ""))
            if msg["role"] == "assistant" and idx in st.session_state.tutorial_results:
                render_tutorial_cards(st.session_state.tutorial_results[idx])

    _cur_step    = _get_onboarding_step(st.session_state.profile)
    _placeholder = (
        "Tell me about your skin..."
        if _cur_step < 5
        else "Ask about skincare, makeup, products... ✨"
    )
    user_input = st.chat_input(_placeholder)

    if user_input:
        step_before = _get_onboarding_step(st.session_state.profile)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            typing_slot = st.empty()
            typing_slot.markdown(
                '<div class="forence-typing">'
                '<div class="dot"></div>'
                '<div class="dot"></div>'
                '<div class="dot"></div>'
                '</div>'
                '<span class="forence-typing-label">Novontis is thinking...</span>',
                unsafe_allow_html=True,
            )

            response_data = get_forence_response(
                user_input=user_input,
                history=st.session_state.messages,
                profile=st.session_state.profile,
                features=st.session_state.features,
                face_b64_chat=st.session_state.face_b64_chat,
                onboarding_step=step_before,
            )

            typing_slot.empty()
            reply   = response_data.get("reply", "Something went wrong. Want to try that again?")
            # Hard sanitize: strip any em/en dashes that GPT produced despite instructions
            reply   = reply.replace("—", ",").replace("–", ",")
            signals = response_data.get("signals", {})

            words = reply.split()
            delay = 0.016 if len(words) > 80 else 0.030
            reply_slot = st.empty()
            displayed  = ""
            for word in words:
                displayed += word + " "
                reply_slot.markdown(displayed + "▌")
                time.sleep(delay)
            reply_slot.markdown(reply)

            updates = signals.get("profile_updates") or {}
            if updates:
                st.session_state.profile = update_profile(st.session_state.profile, updates)

            step_after = _get_onboarding_step(st.session_state.profile)

            # Also treat as completed if GPT confirmed the last question was answered,
            # even if profile_updates was missing (GPT sometimes returns null for preferences)
            last_step_confirmed = (
                step_before == 4
                and signals.get("onboarding_field_collected", False)
            )
            if last_step_confirmed and st.session_state.profile.get("preferences") is None:
                # Force-mark preferences as answered so step resolves to 5
                st.session_state.profile = {**st.session_state.profile, "preferences": "none"}
                step_after = 5

            just_completed_onboarding = (step_before < 5) and (step_after >= 5)

            products_shown: list[dict] = []
            product_reasoning: str     = ""
            onboarding_done = (step_after >= 5)

            should_show_products = (
                onboarding_done
                and (signals.get("show_products", False) or just_completed_onboarding)
                and not signals.get("needs_clarification", False)
                and not signals.get("is_off_topic", False)
            )

            if should_show_products:
                # Build a smart query from profile when onboarding just completed
                if just_completed_onboarding and not signals.get("product_query"):
                    profile_now = st.session_state.profile
                    skin_q      = profile_now.get("skin_type") or "general"
                    concerns_q  = profile_now.get("concerns") or []
                    concern_str = (", ".join(concerns_q) if isinstance(concerns_q, list) else str(concerns_q)) or "skincare routine"
                    product_query = f"{concern_str} for {skin_q} skin"
                else:
                    product_query = signals.get("product_query") or user_input

                product_filters   = signals.get("product_filters") or {}
                product_reasoning = signals.get("product_reasoning", "")

                profile_now     = st.session_state.profile
                allergies_raw   = profile_now.get("allergies") or []
                clean_allergies = [a for a in allergies_raw if a.lower() != "none"]
                if not product_filters.get("avoid_ingredients") and clean_allergies:
                    product_filters["avoid_ingredients"] = clean_allergies
                if not product_filters.get("budget") and profile_now.get("budget"):
                    product_filters["budget"] = profile_now["budget"]

                with st.spinner("Finding the best options for you..."):
                    raw            = search_products_live(query=product_query, filters=product_filters, n=8)
                    products_shown = rank_products(raw, top_n=4)

                if products_shown:
                    render_product_cards(products_shown, reasoning=product_reasoning)
                else:
                    st.caption(
                        "*Couldn't find exact matches right now. "
                        "Try searching the product name directly on Amazon India.*"
                    )

            tutorials_shown: list[dict] = []
            if (
                onboarding_done
                and signals.get("show_tutorials", False)
                and not signals.get("is_off_topic", False)
            ):
                tut_query = signals.get("tutorial_query") or user_input
                with st.spinner("Finding tutorials for you..."):
                    tutorials_shown = search_youtube_tutorials(tut_query, n=3)
                if tutorials_shown:
                    render_tutorial_cards(tutorials_shown)

        assistant_msg_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        if products_shown:
            st.session_state.product_results[assistant_msg_idx] = {
                "products":  products_shown,
                "reasoning": product_reasoning,
            }
        if tutorials_shown:
            st.session_state.tutorial_results[assistant_msg_idx] = tutorials_shown

        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start over with a new photo", type="secondary"):
        for k in _DEFAULTS:
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()