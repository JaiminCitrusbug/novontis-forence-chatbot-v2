"""
Novontis AI — Beauty Assistant with Face Analysis
==================================================
Combines:
  • MediaPipe face-mesh analysis (shape, skin tone, texture)
  • Novontis warm-persona chatbot with structured JSON signals
  • LIVE Amazon product search via SerpAPI (key rotation) + GPT fallback
  • Affiliate waterfall (CJ → Rakuten → Amazon → Google fallback)
  • Silent user-profile builder
  • Clarifying-question & off-topic-redirect logic

Run:
    export OPENAI_API_KEY=sk-...
    export SERPAPI_KEYS=key1,key2,key3
    streamlit run face_analysis.py

NOTE: ChromaDB / generate_kb_v2 are intentionally disabled.
      Products are now fetched live from Amazon at query time.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import io
import json
import math
import re
import time
import uuid

# ── Standard HTTP client (for SerpAPI calls) ──────────────────────────────
import requests

# ── Core imports ──────────────────────────────────────────────────────────
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from openai import OpenAI
from PIL import Image
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# ── ChromaDB — DISABLED (live search replaces KB lookup) ──────────────────
# import chromadb
# from chromadb.utils import embedding_functions
# from generate_kb_v2 import build_kb

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Novontis AI — Your Beauty Bestie",
    page_icon="✨",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# APP-LEVEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL     = "gpt-4o-mini"

# ── SerpAPI key pool ──────────────────────────────────────────────────────
# Set in .env as a comma-separated list:
#   SERPAPI_KEYS=key1,key2,key3
# Each key is a free-tier SerpAPI account (100 searches/month each).
SERPAPI_KEYS: list[str] = [
    k.strip()
    for k in os.getenv("SERPAPI_KEYS", "").split(",")
    if k.strip()
]

# Brand → preferred affiliate network override.
BRAND_AFFILIATE_OVERRIDES: dict[str, str] = {}

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

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

    /* ── Bottom bar ── */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    .stChatFloatingInputContainer,
    .stChatFloatingInputContainer > div {
        background: #faf5f0 !important;
        border-top: 1px solid #ecd8cc !important;
        box-shadow: 0 -4px 20px rgba(180,100,60,0.07) !important;
    }

    /* ── Chat input ── */
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
    [data-testid="stChatInput"] textarea::placeholder {
        color: #c0a090 !important;
    }
    [data-testid="stChatInputSubmitButton"] button,
    .stChatInputContainer button {
        background: linear-gradient(135deg, #c96a40, #e08050) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #fff !important;
    }

    /* ── Chat messages ── */
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
        font-weight: 600 !important;
        color: #5a2a0a !important;
    }

    /* ── Product cards ──
       Each card sits inside its own st.columns() cell.
       Flexbox + fixed min-heights per row keep name / brand / stars / price
       / button aligned at the same vertical position across all 4 columns.
    ── */
    .f-card {
        border: 1px solid #edddd4;
        border-radius: 14px;
        padding: 14px 12px;
        background: #fffaf6;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0;
        box-shadow: 0 2px 8px rgba(160,80,40,0.06);
        height: 100%;
    }
    /* Image slot REMOVED — kept as comment for easy re-enable:
    .f-card img {
        width: 100%;
        height: 130px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    */

    /* Fixed-height zones — every card row occupies the same vertical space */
    .f-card-name {
        font-weight: 700;
        font-size: 0.80rem;
        color: #2b1a1a;
        line-height: 1.35;
        min-height: 52px;        /* room for 3 lines max */
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        width: 100%;
        margin-bottom: 6px;
    }
    .f-card-brand {
        color: #b08060;
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        min-height: 20px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
        margin-bottom: 4px;
    }
    .f-card-stars {
        color: #e8912a;
        font-size: 0.80rem;
        min-height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2px;
        margin-bottom: 4px;
    }
    .f-card-rating-count { color: #aaa; font-size: 0.68rem; }
    .f-card-price {
        font-weight: 800;
        font-size: 0.92rem;
        color: #2b1a1a;
        min-height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .f-card-price-na {
        color: #ccc;
        font-size: 0.75rem;
        min-height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .f-shop-btn {
        display: block;
        background: linear-gradient(135deg, #c96a40, #e08050);
        color: #fff !important;
        text-decoration: none !important;
        padding: 8px 0;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 700;
        transition: opacity .15s;
        width: 100%;
        box-sizing: border-box;
        margin-top: auto;
    }
    .f-shop-btn:hover { opacity: .85; }
    .f-disclosure { font-size: 0.68rem; color: #c0a090; font-style: italic; margin: 2px 0 8px 0; }
    .f-section-label { font-size: 0.92rem; font-weight: 700; color: #a05030; margin: 14px 0 2px 0; }

    /* ── Profile badge ── */
    .profile-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #fff;
        border: 1px solid #edddd4;
        border-radius: 40px;
        padding: 0.5rem 1.1rem;
        font-size: 0.82rem;
        color: #6a3a2a;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(180,90,50,0.09);
        white-space: nowrap;
    }
    .profile-badge .dot { color: #d4a898; margin: 0 2px; }

    /* ── Titles ── */
    h1 {
        font-family: 'DM Serif Display', serif !important;
        color: #2b1a1a !important;
        font-size: 2rem !important;
        letter-spacing: -0.3px;
    }
    .subtitle { font-size: 0.92rem; color: #9a7060; margin-top: 4px; margin-bottom: 1.5rem; }
    .step-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 1px;
        text-transform: uppercase; color: #c07050; margin-bottom: 0.75rem;
    }

    /* ── Buttons ── */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #c96a40, #e08050) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 14px rgba(180,90,50,0.28) !important;
    }
    div.stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #a05030 !important;
        border: 1.5px solid #e0c4b0 !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #fff !important;
        border: 1.5px dashed #e0c4b0 !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }

    [data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 2.5px solid #f0d8c8 !important;
        box-shadow: 0 3px 14px rgba(160,80,40,0.14) !important;
    }

    hr { border-color: #edddd4 !important; }
    .stSpinner > div { color: #c47f5a !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI CLIENT
# ══════════════════════════════════════════════════════════════════════════════

client = OpenAI(api_key=OPENAI_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE — CACHED FACE MESH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY UTILS
# ══════════════════════════════════════════════════════════════════════════════

def to_point(lm, w, h):
    return np.array([int(lm.x * w), int(lm.y * h)])


def distance(a, b):
    return float(np.linalg.norm(a - b))


# ══════════════════════════════════════════════════════════════════════════════
# SKIN REGION SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def sample_skin_region(image, landmarks):
    h, w, _ = image.shape
    pts = [
        to_point(landmarks[10], w, h),
        to_point(landmarks[338], w, h),
        to_point(landmarks[297], w, h),
        to_point(landmarks[332], w, h),
    ]
    x1 = max(min(p[0] for p in pts), 0)
    x2 = min(max(p[0] for p in pts), w)
    y1 = max(min(p[1] for p in pts), 0)
    y2 = min(max(p[1] for p in pts), h)
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        raise ValueError("Unable to sample skin region")
    return region


# ══════════════════════════════════════════════════════════════════════════════
# SKIN COLOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_skin_color(region):
    pixels = region.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    bgr = kmeans.cluster_centers_[0]
    b, g, r = bgr
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    brightness = float((r + g + b) / 3)
    if lab[2] > 150:
        undertone = "warm"
    elif lab[2] < 135:
        undertone = "cool"
    else:
        undertone = "neutral"
    return {
        "rgb": {"r": float(r), "g": float(g), "b": float(b)},
        "lab": {"l": float(lab[0]), "a": float(lab[1]), "b": float(lab[2])},
        "hsv": {"h": float(hsv[0]), "s": float(hsv[1]), "v": float(hsv[2])},
        "brightness": brightness,
        "undertone": undertone,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEXTURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_texture(gray):
    variance = float(np.var(gray))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)
    return {"variance": variance, "edge_density": edge_density}


# ══════════════════════════════════════════════════════════════════════════════
# FACE SHAPE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify_face_shape(width, height, jaw, cheek, forehead):
    h_w = height / width
    cheek_to_jaw = cheek / jaw
    cheek_to_forehead = cheek / forehead
    jaw_to_forehead = jaw / forehead

    if h_w > 1.55 and cheek_to_jaw < 1.20 and cheek_to_forehead < 1.20:
        return "oval"
    if jaw_to_forehead > 0.90 and jaw / cheek > 0.85 and h_w < 1.45:
        return "square"
    if cheek_to_jaw > 1.20 and cheek_to_forehead > 1.15 and h_w >= 1.30:
        return "diamond"
    if h_w < 1.30:
        return "round"
    if forehead / jaw > 1.10:
        return "heart"
    return "oval"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_deep_face_features(pil_image):
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mesh = get_face_mesh()
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        raise ValueError("No face detected")

    lm = results.multi_face_landmarks[0].landmark
    left         = to_point(lm[234], w, h)
    right        = to_point(lm[454], w, h)
    forehead_top = to_point(lm[10],  w, h)
    chin         = to_point(lm[152], w, h)
    jaw_left     = to_point(lm[172], w, h)
    jaw_right    = to_point(lm[397], w, h)
    cheek_left   = to_point(lm[93],  w, h)
    cheek_right  = to_point(lm[323], w, h)
    forehead_left  = to_point(lm[70],  w, h)
    forehead_right = to_point(lm[300], w, h)

    face_width     = distance(left, right)
    face_height    = distance(forehead_top, chin)
    jaw_width      = distance(jaw_left, jaw_right)
    cheek_width    = distance(cheek_left, cheek_right)
    forehead_width = distance(forehead_left, forehead_right)

    face_shape  = classify_face_shape(
        face_width, face_height, jaw_width, cheek_width, forehead_width
    )
    skin_region = sample_skin_region(image, lm)
    skin_color  = analyze_skin_color(skin_region)
    gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture     = analyze_texture(gray)

    return {
        "face_shape": face_shape,
        "facial_dimensions": {
            "face_width_px":      face_width,
            "face_height_px":     face_height,
            "jaw_width_px":       jaw_width,
            "cheekbone_width_px": cheek_width,
            "forehead_width_px":  forehead_width,
            "height_width_ratio": face_height / face_width,
        },
        "skin_color":     skin_color,
        "texture_metrics": texture,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORENCE SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(profile: dict, features: dict | None) -> str:
    face_ctx = ""
    if features:
        shape       = features["face_shape"]
        undertone   = features["skin_color"]["undertone"]
        brightness  = features["skin_color"]["brightness"]
        variance    = features["texture_metrics"]["variance"]
        bright_label = (
            "fair" if brightness > 180 else "medium" if brightness > 100 else "deep"
        )
        face_ctx = (
            f"\n\nFACE ANALYSIS (from the user's uploaded photo — use naturally):\n"
            f"  Face shape: {shape}\n"
            f"  Skin tone: {bright_label} (brightness {brightness:.0f}/255)\n"
            f"  Undertone: {undertone}\n"
            f"  Texture variance: {variance:.1f} (higher = more texture / unevenness)\n"
            f"Weave these facts into your advice when relevant — never announce you are "
            f"reading analysis data."
        )

    profile_ctx = ""
    if profile:
        parts = []
        if profile.get("skin_type"):
            parts.append(f"skin type: {profile['skin_type']}")
        if profile.get("concerns"):
            c = profile["concerns"]
            parts.append(
                "concerns: " + (", ".join(c) if isinstance(c, list) else c)
            )
        if profile.get("preferences"):
            parts.append(f"preferences: {profile['preferences']}")
        if parts:
            profile_ctx = (
                "\n\nKnown user profile (accumulated from this conversation):\n  "
                + "\n  ".join(parts)
                + "\nUse this context naturally — never announce that you are reading a profile."
            )

    return f"""You are Novontis — a warm, knowledgeable, and enthusiastic AI beauty assistant who feels like a best friend that happens to know everything about beauty.
{face_ctx}{profile_ctx}

YOUR PERSONALITY
- Warm, encouraging, never clinical or robotic. Use "you" and "your" naturally.
- Expert across every beauty category: skincare, makeup, haircare, fragrance, body care, self-tanners, nail care, tools.
- Give specific, practical advice tailored to the person — never generic filler.
- If the user's request is vague (e.g. "help me with my skin", "I need something good"), ask ONE smart clarifying question before jumping to recommendations.
- If the user asks about something completely unrelated to beauty (cooking, sports, politics, tech), gently acknowledge it, redirect warmly, and offer to help with beauty instead.
- Silently build an understanding of the user's skin type, concerns, and preferences from everything they share.

RESPONSE FORMAT — MANDATORY
You MUST always respond with a valid JSON object in exactly this structure. No text outside the JSON.

{{
  "reply": "Your warm conversational response here. This is the text shown to the user.",
  "signals": {{
    "show_products": true_or_false,
    "product_query": "a semantic search string describing ideal products (only when show_products is true, else null)",
    "product_filters": {{
      "skin_type": "dry | oily | combination | sensitive | all | null",
      "concerns":  "acne | hydration | anti-aging | oil control | sun protection | dark spots | null",
      "category":  "moisturizer | serum | cleanser | sunscreen | foundation | lipstick | null"
    }},
    "needs_clarification": true_or_false,
    "is_off_topic": true_or_false,
    "profile_updates": {{
      "skin_type":   "detected skin type or null",
      "concerns":    ["concern1", "concern2"] or [],
      "preferences": "e.g. vegan, cruelty-free, fragrance-free or null"
    }}
  }}
}}

RULES FOR show_products
Set to true when:
  • User asks for a product recommendation, what to buy, or what to use
  • User asks for a dupe, alternative, or replacement for any product
  • A recommendation is clearly useful in context
Set to false when:
  • You are asking a clarifying question (also set needs_clarification: true)
  • Redirecting an off-topic message (also set is_off_topic: true)
  • Giving general how-to technique advice without a product ask
  • Greeting or small talk

PRODUCT FILTER RULES
Only populate a filter field if you are confident about the value.
Use null for anything uncertain — a broader search is better than a wrong filter.

REPLY FORMATTING RULES
Your "reply" text is rendered as Markdown inside a Streamlit chat bubble.
- Use **bold** for emphasis (product names, key terms).
- Use numbered lists (1. 2. 3.) or bullet points (- ) when giving multi-step advice or listing items.
- Add a blank line between paragraphs and before/after lists for readability.
- Keep each point concise — 1-2 sentences max.
- Never dump everything into one giant paragraph.
"""


# ══════════════════════════════════════════════════════════════════════════════
# FORENCE RESPONSE — calls GPT-4o-mini, returns structured dict
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_SIGNALS = {
    "show_products":      False,
    "product_query":      None,
    "product_filters":    {},
    "needs_clarification": False,
    "is_off_topic":       False,
    "profile_updates":    {},
}


def get_forence_response(
    user_input: str,
    history:    list[dict],
    profile:    dict,
    features:   dict | None,
) -> dict:
    messages = [
        {"role": "system", "content": build_system_prompt(profile, features)}
    ]
    for msg in history[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    raw = ""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.72,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        raw    = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)

        parsed.setdefault(
            "reply", "I'm here for you! How can I help with your beauty routine?"
        )
        parsed.setdefault("signals", _FALLBACK_SIGNALS)
        for k, v in _FALLBACK_SIGNALS.items():
            parsed["signals"].setdefault(k, v)

        return parsed

    except json.JSONDecodeError:
        return {
            "reply":   raw or "Something went wrong — please try again!",
            "signals": _FALLBACK_SIGNALS,
        }
    except Exception as e:
        return {
            "reply":   f"Something went wrong on my end — please try again! ({e})",
            "signals": _FALLBACK_SIGNALS,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ── LIVE PRODUCT SEARCH ENGINE ────────────────────────────────────────────────
# Architecture:
#   1. SerpAPI key rotation  — tries each key in pool, skips exhausted ones
#   2. GPT-4o-mini fallback  — if ALL SerpAPI keys fail, GPT returns real
#                              brand names + direct Amazon product page URLs
# ══════════════════════════════════════════════════════════════════════════════

# ── Known beauty brands for title→brand extraction ────────────────────────
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
    """Match known brand name inside title, else capitalise first word."""
    title_lower = title.lower()
    for brand in _KNOWN_BRANDS:
        if brand.lower() in title_lower:
            return brand
    return title.split()[0].capitalize() if title else "Unknown"


def _parse_price(raw) -> float:
    """'₹1,299' / '$24.99' / 24.99 → float."""
    if not raw:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return 0.0


def _parse_rating(raw) -> float:
    """'4.5 out of 5 stars' / 4.5 → float."""
    if not raw:
        return 0.0
    m = re.search(r"(\d+\.?\d*)", str(raw))
    return float(m.group(1)) if m else 0.0


def _parse_count(raw) -> int:
    """'1,234 ratings' / '1234' → int."""
    if not raw:
        return 0
    cleaned = re.sub(r"[^\d]", "", str(raw))
    try:
        return int(cleaned)
    except ValueError:
        return 0


def _extract_asin(url: str) -> str | None:
    """Extract a 10-character ASIN from any Amazon URL string."""
    m = re.search(r"/(?:dp|gp/product|product)/([A-Z0-9]{10})", url)
    return m.group(1) if m else None


def _build_amazon_product_url(raw_link: str, title: str) -> str:
    """
    CHANGE 1 — Return a direct Amazon PRODUCT PAGE url, not a search URL.

    Priority order:
      1. If raw_link already contains /dp/ASIN  →  clean canonical product URL
      2. If raw_link contains /gp/product/ASIN  →  convert to /dp/ASIN
      3. If raw_link contains an ASIN anywhere  →  build /dp/ASIN URL
      4. If raw_link is any other Amazon URL    →  return it as-is (still a
         product page in most cases — SerpAPI usually returns product links)
      5. Last resort: Amazon India keyword SEARCH url
         (shown only when SerpAPI gives no link at all)
    """
    if raw_link and "amazon" in raw_link:
        # Try to find /dp/ASIN or /gp/product/ASIN
        asin = _extract_asin(raw_link)
        if asin:
            # Normalise to the clean canonical form
            return f"https://www.amazon.in/dp/{asin}"
        # No ASIN found but it's still an Amazon URL — return as-is
        # (SerpAPI product links usually go straight to the product page)
        return raw_link

    # No Amazon link at all → fall back to keyword search
    q = re.sub(r"\s+", "+", title.strip())
    return f"https://www.amazon.in/s?k={q}"


def _make_product_dict(
    title:     str,
    image_url: str,
    price_raw,
    rating_raw,
    count_raw,
    amazon_url: str,
    category:   str = "",
    skin_type:  str = "all",
    concerns:   str = "",
    distance:   float = 0.5,
) -> dict:
    """
    Assemble a product dict that matches the schema expected by
    rank_products() and render_product_cards() exactly.
    """
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
        # Affiliate columns — direct Amazon product URL stored in affiliate_amazon
        "affiliate_cj":      "",
        "affiliate_rakuten": "",
        "affiliate_amazon":  amazon_url,
        "image_url":         image_url or "",
        # Ranking support
        "_distance":         distance,
    }


# ── SerpAPI key rotation ──────────────────────────────────────────────────

def _get_serpapi_key_pool() -> list[str]:
    if "serpapi_key_pool" not in st.session_state:
        st.session_state.serpapi_key_pool  = list(SERPAPI_KEYS)
        st.session_state.serpapi_key_index = 0
    return st.session_state.serpapi_key_pool


def _current_serpapi_key() -> str | None:
    pool = _get_serpapi_key_pool()
    if not pool:
        return None
    idx = st.session_state.get("serpapi_key_index", 0) % len(pool)
    return pool[idx]


def _rotate_serpapi_key() -> str | None:
    pool = _get_serpapi_key_pool()
    if not pool:
        return None
    current = st.session_state.get("serpapi_key_index", 0)
    next_idx = current + 1
    if next_idx >= len(pool):
        return None
    st.session_state.serpapi_key_index = next_idx
    return pool[next_idx]


def _reset_serpapi_rotation():
    st.session_state.serpapi_key_index = 0


# ── Single SerpAPI call ───────────────────────────────────────────────────

_SERPAPI_QUOTA_CODES = {429, 401, 403}


def _serpapi_search(query: str, api_key: str, n: int = 5) -> list[dict]:
    params = {
        "engine":        "amazon",
        "q":             query,
        "api_key":       api_key,
        "amazon_domain": "amazon.in",
    }
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=12,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if resp.status_code in _SERPAPI_QUOTA_CODES:
        raise PermissionError(
            f"SerpAPI key exhausted or invalid (HTTP {resp.status_code})"
        )
    if not resp.ok:
        raise RuntimeError(f"SerpAPI HTTP {resp.status_code}")

    data    = resp.json()
    organic = data.get("organic_results") or data.get("results") or []
    return organic[:n]


def _serpapi_results_to_products(
    organic: list[dict],
    category: str,
    skin_type: str,
    concerns: str,
) -> list[dict]:
    """Convert raw SerpAPI organic results into Novontis product dicts."""
    products = []
    for i, item in enumerate(organic):
        title = (item.get("title") or "").strip()
        if not title:
            continue

        # CHANGE 1 — use _build_amazon_product_url (direct product page)
        raw_link   = item.get("link") or item.get("product_link") or ""
        amazon_url = _build_amazon_product_url(raw_link, title)

        # image_url kept in dict for future use but not rendered (CHANGE 2)
        image_url  = item.get("thumbnail") or item.get("image") or ""

        price_raw = (
            item.get("price")
            or ((item.get("prices") or [{}])[0].get("raw", ""))
        )
        rating_raw = item.get("rating")
        count_raw  = item.get("reviews_count") or item.get("ratings_total")

        products.append(
            _make_product_dict(
                title=title,
                image_url=image_url,
                price_raw=price_raw,
                rating_raw=rating_raw,
                count_raw=count_raw,
                amazon_url=amazon_url,
                category=category,
                skin_type=skin_type,
                concerns=concerns,
                distance=float(i) / max(len(organic), 1),
            )
        )
    return products


# ── GPT fallback ──────────────────────────────────────────────────────────

# CHANGE 1 — updated GPT system prompt to request direct /dp/ASIN URLs
_GPT_PRODUCT_SYSTEM = """You are a beauty product data assistant.
When given a search query, return a JSON array of real beauty products
available on Amazon India. Each element must have exactly these fields:
  name, brand, category, price_inr (number, 0 if unknown),
  rating (number 0–5), review_count (integer),
  asin (10-character Amazon ASIN string, e.g. "B08XYZ1234" — required if you know it),
  amazon_product_url (direct Amazon India product page URL in the format
    https://www.amazon.in/dp/<ASIN>  — use the ASIN field to build this;
    if ASIN is unknown use https://www.amazon.in/s?k=URL-encoded+product+name)

Return ONLY the JSON array. No markdown, no explanation, no extra keys.
Limit to {n} products. Use real brand names and real product names only."""


def _gpt_product_fallback(
    query:     str,
    filters:   dict,
    n:         int = 5,
) -> list[dict]:
    """
    Ask GPT-4o-mini to suggest real beauty products when SerpAPI is unavailable.
    Returns product dicts with direct Amazon product page URLs where possible.
    """
    category  = filters.get("category")  or ""
    skin_type = filters.get("skin_type") or "all"
    concerns  = filters.get("concerns")  or ""

    filter_hint = ""
    if category:
        filter_hint += f" Category: {category}."
    if skin_type and skin_type not in ("all", "null"):
        filter_hint += f" For {skin_type} skin."
    if concerns and concerns != "null":
        filter_hint += f" Target concern: {concerns}."

    user_msg = (
        f"Search query: {query}{filter_hint}\n"
        f"Return {n} real Amazon India beauty products as a JSON array."
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _GPT_PRODUCT_SYSTEM.format(n=n),
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw  = resp.choices[0].message.content.strip()
        data = json.loads(raw)

        # GPT sometimes wraps the array under a key
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                return []

        products = []
        for i, item in enumerate(data[:n]):
            title     = str(item.get("name", "")).strip()
            brand     = str(item.get("brand", "")).strip() or _extract_brand(title)
            cat       = str(item.get("category", category)).strip()
            price_raw = item.get("price_inr", 0)
            rating    = float(item.get("rating", 0) or 0)
            count     = int(item.get("review_count", 0) or 0)

            # CHANGE 1 — prefer /dp/ASIN URL provided by GPT
            asin = str(item.get("asin", "")).strip()
            url  = str(item.get("amazon_product_url", "")).strip()

            if asin and re.fullmatch(r"[A-Z0-9]{10}", asin):
                url = f"https://www.amazon.in/dp/{asin}"
            elif not url:
                q   = re.sub(r"\s+", "+", f"{title} {brand}".strip())
                url = f"https://www.amazon.in/s?k={q}"

            if not title:
                continue

            p = _make_product_dict(
                title=title,
                image_url="",           # GPT doesn't give images
                price_raw=price_raw,
                rating_raw=rating,
                count_raw=count,
                amazon_url=url,
                category=cat,
                skin_type=skin_type,
                concerns=concerns,
                distance=float(i) / max(n, 1),
            )
            p["brand"] = brand
            products.append(p)

        return products

    except Exception as e:
        print(f"[GPT fallback error] {e}")
        return []


# ── Main orchestrator ─────────────────────────────────────────────────────

def search_products_live(
    query:   str,
    filters: dict,
    n:       int = 8,
) -> list[dict]:
    """
    Fetch real Amazon products for the given query.

    Flow:
      1. Try each SerpAPI key in the pool (rotation on quota/error).
      2. If all SerpAPI keys fail → GPT-4o-mini fallback.
      3. Returns list[dict] in the same schema as the old query_products().
    """
    category  = (filters.get("category")  or "").strip()
    skin_type = (filters.get("skin_type") or "all").strip()
    concerns  = (filters.get("concerns")  or "").strip()

    enriched_query = query
    if category and category not in ("null", ""):
        enriched_query = f"{category} {enriched_query}"
    if skin_type and skin_type not in ("null", "all", ""):
        enriched_query += f" for {skin_type} skin"
    if concerns and concerns not in ("null", ""):
        enriched_query += f" {concerns}"

    # ── SerpAPI with key rotation ─────────────────────────────────────────
    if SERPAPI_KEYS:
        _reset_serpapi_rotation()

        while True:
            api_key = _current_serpapi_key()
            if api_key is None:
                print("[SerpAPI] All keys exhausted — falling back to GPT")
                break

            print(f"[SerpAPI] Trying key ...{api_key[-6:]} | query: {enriched_query!r}")
            try:
                organic  = _serpapi_search(enriched_query, api_key, n=n)
                products = _serpapi_results_to_products(
                    organic, category, skin_type, concerns
                )
                if products:
                    print(f"[SerpAPI] Got {len(products)} products")
                    return products
                else:
                    print(f"[SerpAPI] Zero results — rotating key")
                    _rotate_serpapi_key()

            except PermissionError as e:
                print(f"[SerpAPI] Key invalid/exhausted: {e} — rotating")
                next_key = _rotate_serpapi_key()
                if next_key is None:
                    print("[SerpAPI] No more keys — falling back to GPT")
                    break

            except Exception as e:
                print(f"[SerpAPI] Error: {e} — rotating key")
                next_key = _rotate_serpapi_key()
                if next_key is None:
                    print("[SerpAPI] No more keys — falling back to GPT")
                    break

    else:
        print("[SerpAPI] No keys configured — using GPT fallback")

    # ── GPT fallback ──────────────────────────────────────────────────────
    print(f"[GPT fallback] Fetching products for: {query!r}")
    products = _gpt_product_fallback(query, filters, n=min(n, 6))
    print(f"[GPT fallback] Got {len(products)} products")
    return products


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT RANKING
# ══════════════════════════════════════════════════════════════════════════════

def rank_products(products: list[dict], top_n: int = 4) -> list[dict]:
    for p in products:
        rating  = float(p.get("rating", 0) or 0)
        count   = float(p.get("rating_count", 0) or 0)
        dist    = float(p.get("_distance", 1.0))
        quality   = rating * math.log1p(count)
        relevance = 1.0 / (1.0 + dist)
        p["_score"] = 0.4 * relevance + 0.6 * (quality / 50.0)
    return sorted(products, key=lambda x: x["_score"], reverse=True)[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# AFFILIATE WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

def get_affiliate_link(product: dict) -> tuple[str, str]:
    brand    = product.get("brand", "").lower().strip()
    override = BRAND_AFFILIATE_OVERRIDES.get(brand)

    if override == "cj"      and product.get("affiliate_cj"):
        return product["affiliate_cj"],      "CJ"
    if override == "rakuten" and product.get("affiliate_rakuten"):
        return product["affiliate_rakuten"], "Rakuten"
    if override == "amazon"  and product.get("affiliate_amazon"):
        return product["affiliate_amazon"],  "Amazon"

    if product.get("affiliate_cj"):
        return product["affiliate_cj"],      "CJ"
    if product.get("affiliate_rakuten"):
        return product["affiliate_rakuten"], "Rakuten"
    if product.get("affiliate_amazon"):
        return product["affiliate_amazon"],  "Amazon"

    q = (
        (product.get("name", "") + " " + product.get("brand", ""))
        .strip()
        .replace(" ", "+")
    )
    return f"https://www.google.com/search?q={q}", "web"


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT CARD RENDERER
# CHANGE 2 — product image is commented out
# CHANGE 3 — CSS grid rows ensure all 4 cards are perfectly aligned
# ══════════════════════════════════════════════════════════════════════════════

def _star_html(rating: float) -> str:
    filled = round(rating)
    return "★" * filled + "☆" * (5 - filled)


def render_product_cards(products: list[dict]):
    if not products:
        return

    has_affiliate = any(
        p.get("affiliate_cj") or p.get("affiliate_rakuten") or p.get("affiliate_amazon")
        for p in products
    )

    st.markdown(
        "<div class='f-section-label'>🛍️ Products Novontis thinks you'll love</div>",
        unsafe_allow_html=True,
    )
    if has_affiliate:
        st.markdown(
            "<p class='f-disclosure'>* Links open the Amazon product page directly.</p>",
            unsafe_allow_html=True,
        )

    # Use st.columns() for layout — Streamlit renders this correctly inside
    # st.chat_message, unlike a single large HTML block which gets escaped.
    # Each card is its own st.markdown() call so HTML is always rendered.
    cols = st.columns(min(len(products), 4))

    for col, product in zip(cols, products):
        affiliate_url, _ = get_affiliate_link(product)
        rating       = float(product.get("rating", 0) or 0)
        rating_count = int(product.get("rating_count", 0) or 0)
        price        = float(product.get("price", 0) or 0)
        name         = product.get("name", "Unknown Product")
        brand        = product.get("brand", "")

        # ── Image tag commented out (CHANGE 2) ────────────────────────────
        # image_url = product.get("image_url") or ""
        # if not image_url or "placeholder" in image_url.lower():
        #     image_url = PLACEHOLDER_IMAGE
        # img_html = (
        #     f'<img src="{image_url}" '
        #     f'onerror="this.onerror=null;this.src=\'{PLACEHOLDER_IMAGE}\'" />'
        # )
        # ── End commented-out image block ─────────────────────────────────

        if rating > 0:
            rating_html = (
                f"<div class='f-card-stars'>{_star_html(rating)}&nbsp;"
                f"<span style='font-size:.76rem;color:#666'>{rating:.1f}</span>"
                f"<span class='f-card-rating-count'>&nbsp;({rating_count:,})</span></div>"
            )
        else:
            rating_html = (
                "<div class='f-card-stars' style='color:#ddd'>☆☆☆☆☆ "
                "<span style='font-size:.7rem;color:#ccc'>not rated</span></div>"
            )

        price_html = (
            f"<div class='f-card-price'>₹{price:,.0f}</div>"
            if price > 0
            else "<div class='f-card-price-na'>Price unavailable</div>"
        )

        # Each card is injected individually per column — no cross-column HTML
        with col:
            st.markdown(
                f"""
                <div class="f-card">
                    <div class="f-card-name">{name}</div>
                    <div class="f-card-brand">{brand}</div>
                    {rating_html}
                    {price_html}
                    <a href="{affiliate_url}" target="_blank" class="f-shop-btn">Shop Now →</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE UPDATER
# ══════════════════════════════════════════════════════════════════════════════

def update_profile(profile: dict, updates: dict) -> dict:
    if not updates:
        return profile
    profile = profile.copy()
    if updates.get("skin_type"):
        profile["skin_type"] = updates["skin_type"]
    raw_concerns = updates.get("concerns", [])
    if raw_concerns:
        if isinstance(raw_concerns, str):
            raw_concerns = [raw_concerns]
        existing = profile.get("concerns", [])
        if isinstance(existing, str):
            existing = [existing]
        profile["concerns"] = sorted(set(existing + raw_concerns))
    if updates.get("preferences"):
        profile["preferences"] = updates["preferences"]
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "features":        None,
    "messages":        [],
    "analyzed":        False,
    "face_image_bytes": None,
    "profile":         {},
    "product_results": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<h1>✨ Novontis — Your Beauty Bestie</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Personalised makeup & skincare advice — '
    "powered by your face analysis.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — UPLOAD & ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.analyzed:

    st.markdown(
        '<p class="step-label">Step 1 of 2 · Upload your photo</p>',
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.error(
            "⚠️ **OPENAI_API_KEY not set.** "
            "Add it to your .env: `OPENAI_API_KEY=sk-...`"
        )

    # if not SERPAPI_KEYS:
    #     st.info(
    #         "ℹ️ **SERPAPI_KEYS not set** — product search will use GPT fallback.\n\n"
    #         "For richer results, add free SerpAPI keys to your .env:\n"
    #         "`SERPAPI_KEYS=key1,key2,key3`\n\n"
    #         "Get free keys at [serpapi.com](https://serpapi.com) "
    #         "(100 searches/month each)."
    #     )

    uploaded_file = st.file_uploader(
        "Choose a clear, well-lit face photo",
        type=["jpg", "jpeg", "png"],
        help="Front-facing photos with good lighting give the best results.",
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file)

        thumb = pil_image.copy()
        thumb.thumbnail((300, 220), Image.LANCZOS)
        _, col_c, _ = st.columns([1, 2, 1])
        with col_c:
            st.image(thumb, use_column_width=True, caption="Your uploaded photo")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Analyse My Face & Start Chatting →",
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("Scanning your features — hang tight…"):
                try:
                    features = extract_deep_face_features(pil_image)
                    st.session_state.features = features
                    st.session_state.analyzed = True

                    avatar = pil_image.copy()
                    avatar.thumbnail((120, 120), Image.LANCZOS)
                    buf = io.BytesIO()
                    avatar.save(buf, format="PNG")
                    st.session_state.face_image_bytes = buf.getvalue()

                    f               = features
                    shape           = f["face_shape"]
                    undertone       = f["skin_color"]["undertone"]
                    brightness_label = (
                        "fair"   if f["skin_color"]["brightness"] > 180 else
                        "medium" if f["skin_color"]["brightness"] > 100 else
                        "deep"
                    )
                    opener = (
                        f"Great news — I've analysed your photo! 🎉\n\n"
                        f"Here's what I found:\n"
                        f"- **Face shape:** {shape.capitalize()}\n"
                        f"- **Skin tone:** {brightness_label.capitalize()} with "
                        f"a **{undertone}** undertone\n\n"
                        f"Ask me anything — makeup tips, product recommendations, "
                        f"contouring tricks, skincare routines… I'm here to help!"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": opener}
                    )
                    st.rerun()

                except Exception as e:
                    st.error(
                        f"⚠️ Face detection failed: {e}\n\n"
                        "Please try a clearer, front-facing photo with good lighting."
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CHAT + PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

else:

    st.markdown(
        '<p class="step-label">Step 2 of 2 · Your beauty assistant</p>',
        unsafe_allow_html=True,
    )

    f          = st.session_state.features
    shape      = f["face_shape"].capitalize()
    undertone  = f["skin_color"]["undertone"].capitalize()
    brightness = f["skin_color"]["brightness"]
    tone_label = "Fair" if brightness > 180 else "Medium" if brightness > 100 else "Deep"

    img_col, badge_col = st.columns([1, 4], gap="small")
    with img_col:
        if st.session_state.face_image_bytes:
            st.image(
                io.BytesIO(st.session_state.face_image_bytes),
                use_column_width=True,
                output_format="PNG",
            )
    with badge_col:
        badge_parts = [
            f"🪞 <strong>{shape}</strong> face",
            f"<strong>{tone_label}</strong> skin",
            f"<strong>{undertone}</strong> undertone",
        ]
        p = st.session_state.profile
        if p.get("skin_type"):
            badge_parts.append(f"🌿 {p['skin_type']} skin")
        if p.get("concerns"):
            c = p["concerns"]
            badge_parts.append(
                "💡 " + (", ".join(c) if isinstance(c, list) else c)
            )

        st.markdown(
            '<div class="profile-badge" style="margin-top:8px;">'
            + '<span class="dot"> · </span>'.join(badge_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Render chat history ───────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and idx in st.session_state.product_results
            ):
                render_product_cards(st.session_state.product_results[idx])

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about makeup, skincare, products… ✨")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Novontis is thinking…"):
                response_data = get_forence_response(
                    user_input=user_input,
                    history=st.session_state.messages,
                    profile=st.session_state.profile,
                    features=st.session_state.features,
                )

            reply   = response_data.get("reply", "Sorry, something went wrong — try again!")
            signals = response_data.get("signals", {})

            st.markdown(reply)

            updates = signals.get("profile_updates", {})
            if updates:
                st.session_state.profile = update_profile(
                    st.session_state.profile, updates
                )

            products_shown: list[dict] = []

            should_show = (
                signals.get("show_products", False)
                and not signals.get("needs_clarification", False)
                and not signals.get("is_off_topic", False)
            )

            if should_show:
                product_query   = signals.get("product_query") or user_input
                product_filters = signals.get("product_filters") or {}

                with st.spinner("Finding the best products for you on Amazon…"):
                    raw            = search_products_live(
                        query=product_query,
                        filters=product_filters,
                        n=8,
                    )
                    products_shown = rank_products(raw, top_n=4)

                if products_shown:
                    render_product_cards(products_shown)
                else:
                    st.caption(
                        "*I couldn't find matching products right now — "
                        "try rephrasing your question!*"
                    )

        assistant_msg_idx = len(st.session_state.messages)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        if products_shown:
            st.session_state.product_results[assistant_msg_idx] = products_shown
        st.rerun()

    # ── Reset button ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start over with a new photo", type="secondary"):
        for k in _DEFAULTS:
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()