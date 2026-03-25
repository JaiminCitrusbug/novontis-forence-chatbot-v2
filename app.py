"""
Forence AI — Beauty Assistant with Face Analysis
==================================================
Combines:
  • MediaPipe face-mesh analysis (shape, skin tone, texture)
  • Forence warm-persona chatbot with structured JSON signals
  • ChromaDB semantic product search + ranked product cards
  • Affiliate waterfall (CJ → Rakuten → Amazon → Google fallback)
  • Silent user-profile builder
  • Clarifying-question & off-topic-redirect logic

Run:
    export OPENAI_API_KEY=sk-...
    python generate_kb_v2.py          # build knowledge base first
    streamlit run face_analysis.py
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

import chromadb
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from chromadb.utils import embedding_functions
from openai import OpenAI
from PIL import Image
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from generate_kb_v2 import build_kb
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Forence AI — Your Beauty Bestie",
    page_icon="✨",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# APP-LEVEL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH       = "chroma_db"  # must match generate_kb_v2.py
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
CHAT_MODEL        = "gpt-4o-mini"
EMBEDDING_MODEL   = "text-embedding-3-large"

# ══════════════════════════════════════════════════════════════════════════════
# AUTO BUILD KNOWLEDGE BASE (STREAMLIT SAFE)
# ══════════════════════════════════════════════════════════════════════════════

def ensure_kb_exists():
    """
    Ensures ChromaDB exists.
    If collections are missing, builds the KB automatically.

    Safe for:
    - Streamlit Cloud
    - Docker
    - Local
    - CI/CD
    """

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY missing — cannot build KB")
        return False

    try:

        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )

        chroma = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        # Verify collections exist
        chroma.get_collection(
            name="products",
            embedding_function=embed_fn,
        )

        chroma.get_collection(
            name="videos",
            embedding_function=embed_fn,
        )

        print("Knowledge base already exists")

        return True

    except Exception:

        print("Knowledge base missing — building now...")

        try:

            build_kb()

            print("Knowledge base successfully created")

            return True

        except Exception as e:

            print("KB build failed:", e)

            return False


@st.cache_resource
def initialize_kb():

    return ensure_kb_exists()

initialize_kb()
# Base64 inline SVG — always loads, no external dependency
PLACEHOLDER_IMAGE = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMu"
    "b3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgZmlsbD0i"
    "I2YwZTBkNCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBkb21pbmFudC1iYXNlbGluZT0i"
    "bWlkZGxlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0ic2Fucy1zZXJp"
    "ZiIgZm9udC1zaXplPSIxMiIgZmlsbD0iIzlhNzA2MCI+Tm8gSW1hZ2U8L3RleHQ+PC9z"
    "dmc+"
)

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

    /* ── Product cards ── */
    .f-card {
        border: 1px solid #edddd4;
        border-radius: 14px;
        padding: 14px 12px;
        background: #fffaf6;
        text-align: center;
        min-height: 340px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 8px rgba(160,80,40,0.06);
        margin-bottom: 8px;
    }
    .f-card img {
        width: 100%;
        height: 130px;
        object-fit: cover;
        border-radius: 10px;
    }
    .f-card-name  { font-weight: 700; font-size: 0.82rem; color: #2b1a1a; line-height: 1.35; }
    .f-card-brand { color: #b08060; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .f-card-stars { color: #e8912a; font-size: 0.82rem; margin: 2px 0; }
    .f-card-rating-count { color: #aaa; font-size: 0.68rem; }
    .f-card-price { font-weight: 800; font-size: 0.95rem; color: #2b1a1a; }
    .f-card-price-na { color: #ccc; font-size: 0.78rem; }
    .f-shop-btn {
        display: inline-block;
        background: linear-gradient(135deg, #c96a40, #e08050);
        color: #fff !important;
        text-decoration: none !important;
        padding: 7px 20px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: auto;
        transition: opacity .15s;
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
# CHROMADB — CACHED PRODUCT COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_chroma_collections():
    """
    Returns (products_collection, videos_collection, kb_available).
    Safe to call even if the KB hasn't been built yet.
    """
    if not OPENAI_API_KEY:
        return None, None, False
    try:
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )
        chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        products = chroma.get_collection(name="products", embedding_function=embed_fn)
        videos = chroma.get_collection(name="videos", embedding_function=embed_fn)
        return products, videos, True
    except Exception:
        return None, None, False


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
    left = to_point(lm[234], w, h)
    right = to_point(lm[454], w, h)
    forehead_top = to_point(lm[10], w, h)
    chin = to_point(lm[152], w, h)
    jaw_left = to_point(lm[172], w, h)
    jaw_right = to_point(lm[397], w, h)
    cheek_left = to_point(lm[93], w, h)
    cheek_right = to_point(lm[323], w, h)
    forehead_left = to_point(lm[70], w, h)
    forehead_right = to_point(lm[300], w, h)

    face_width = distance(left, right)
    face_height = distance(forehead_top, chin)
    jaw_width = distance(jaw_left, jaw_right)
    cheek_width = distance(cheek_left, cheek_right)
    forehead_width = distance(forehead_left, forehead_right)

    face_shape = classify_face_shape(
        face_width, face_height, jaw_width, cheek_width, forehead_width
    )
    skin_region = sample_skin_region(image, lm)
    skin_color = analyze_skin_color(skin_region)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = analyze_texture(gray)

    return {
        "face_shape": face_shape,
        "facial_dimensions": {
            "face_width_px": face_width,
            "face_height_px": face_height,
            "jaw_width_px": jaw_width,
            "cheekbone_width_px": cheek_width,
            "forehead_width_px": forehead_width,
            "height_width_ratio": face_height / face_width,
        },
        "skin_color": skin_color,
        "texture_metrics": texture,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORENCE SYSTEM PROMPT — with face features + user profile
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(profile: dict, features: dict | None) -> str:
    """
    Builds the Forence persona prompt, injecting:
      • face-analysis results (shape, undertone, brightness, texture)
      • accumulated user profile (skin type, concerns, preferences)
    """

    # ── Face-analysis context ────────────────────────────────────────────
    face_ctx = ""
    if features:
        shape = features["face_shape"]
        undertone = features["skin_color"]["undertone"]
        brightness = features["skin_color"]["brightness"]
        variance = features["texture_metrics"]["variance"]
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

    # ── Silent user profile ──────────────────────────────────────────────
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

    return f"""You are Forence — a warm, knowledgeable, and enthusiastic AI beauty assistant who feels like a best friend that happens to know everything about beauty.
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
    "show_products": False,
    "product_query": None,
    "product_filters": {},
    "needs_clarification": False,
    "is_off_topic": False,
    "profile_updates": {},
}


def get_forence_response(
    user_input: str,
    history: list[dict],
    profile: dict,
    features: dict | None,
) -> dict:
    """
    Returns {'reply': str, 'signals': dict}.
    Always safe — returns fallback on any error.
    """
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
        raw = resp.choices[0].message.content.strip()
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
            "reply": raw or "Something went wrong — please try again!",
            "signals": _FALLBACK_SIGNALS,
        }
    except Exception as e:
        return {
            "reply": f"Something went wrong on my end — please try again! ({e})",
            "signals": _FALLBACK_SIGNALS,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT SEARCH  (ChromaDB semantic query)
# ══════════════════════════════════════════════════════════════════════════════

def query_products(query: str, filters: dict, n_results: int = 12) -> list[dict]:
    products_col, _, kb_ok = get_chroma_collections()
    if not kb_ok or products_col is None:
        return []

    def run_query(where: dict | None) -> list[dict]:
        kwargs: dict = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        try:
            res = products_col.query(**kwargs)
        except Exception:
            return []
        if not res.get("metadatas") or not res["metadatas"][0]:
            return []
        out = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            meta["_distance"] = float(dist)
            out.append(meta)
        return out

    # Build where-clause
    conditions = []
    if filters:
        skin = filters.get("skin_type")
        if skin and skin not in (None, "null", "all"):
            conditions.append({"skin_type": {"$in": [skin, "all"]}})
        concern = filters.get("concerns")
        if concern and concern not in (None, "null"):
            conditions.append({"concerns": {"$eq": concern}})
        category = filters.get("category")
        if category and category not in (None, "null"):
            conditions.append({"category": {"$eq": category}})

    where_clause = (
        None
        if len(conditions) == 0
        else conditions[0]
        if len(conditions) == 1
        else {"$and": conditions}
    )

    results = run_query(where_clause)
    if not results and where_clause is not None:
        results = run_query(None)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# AFFILIATE WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

def get_affiliate_link(product: dict) -> tuple[str, str]:
    brand = product.get("brand", "").lower().strip()
    override = BRAND_AFFILIATE_OVERRIDES.get(brand)

    if override == "cj" and product.get("affiliate_cj"):
        return product["affiliate_cj"], "CJ"
    if override == "rakuten" and product.get("affiliate_rakuten"):
        return product["affiliate_rakuten"], "Rakuten"
    if override == "amazon" and product.get("affiliate_amazon"):
        return product["affiliate_amazon"], "Amazon"

    if product.get("affiliate_cj"):
        return product["affiliate_cj"], "CJ"
    if product.get("affiliate_rakuten"):
        return product["affiliate_rakuten"], "Rakuten"
    if product.get("affiliate_amazon"):
        return product["affiliate_amazon"], "Amazon"

    q = (
        (product.get("name", "") + " " + product.get("brand", ""))
        .strip()
        .replace(" ", "+")
    )
    return f"https://www.google.com/search?q={q}", "web"


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT RANKING
# ══════════════════════════════════════════════════════════════════════════════

def rank_products(products: list[dict], top_n: int = 4) -> list[dict]:
    for p in products:
        rating = float(p.get("rating", 0) or 0)
        count = float(p.get("rating_count", 0) or 0)
        dist = float(p.get("_distance", 1.0))
        quality = rating * math.log1p(count)
        relevance = 1.0 / (1.0 + dist)
        p["_score"] = 0.4 * relevance + 0.6 * (quality / 50.0)
    return sorted(products, key=lambda x: x["_score"], reverse=True)[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT CARD RENDERER
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
        "<div class='f-section-label'>🛍️ Products Forence thinks you'll love</div>",
        unsafe_allow_html=True,
    )
    if has_affiliate:
        st.markdown(
            "<p class='f-disclosure'>* Some links are affiliate links — "
            "Forence may earn a small commission at no extra cost to you.</p>",
            unsafe_allow_html=True,
        )

    cols = st.columns(min(len(products), 4))
    for col, product in zip(cols, products):
        with col:
            affiliate_url, _ = get_affiliate_link(product)
            rating = float(product.get("rating", 0) or 0)
            rating_count = int(product.get("rating_count", 0) or 0)
            price = float(product.get("price", 0) or 0)
            raw_img = product.get("image_url") or ""
            # Catch old placeholder URLs from previously-built KBs
            if not raw_img or "placeholder" in raw_img.lower():
                image_url = PLACEHOLDER_IMAGE
            else:
                image_url = raw_img
            name = product.get("name", "Unknown Product")
            brand = product.get("brand", "")

            if rating > 0:
                rating_html = (
                    f"<div class='f-card-stars'>{_star_html(rating)}&nbsp;"
                    f"<span style='font-size:.78rem;color:#666'>{rating:.1f}</span>"
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

            st.markdown(
                f"""
                <div class="f-card">
                    <img src="{image_url}"
                         onerror="this.onerror=null;this.src='{PLACEHOLDER_IMAGE}'" />
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
    "features": None,
    "messages": [],
    "analyzed": False,
    "face_image_bytes": None,
    "profile": {},
    "product_results": {},  # assistant-message-index → list[dict]
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<h1>✨ Forence — Your Beauty Bestie</h1>", unsafe_allow_html=True)
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

    # KB status (small inline notice)
    _, _, kb_ok = get_chroma_collections()
    if not OPENAI_API_KEY:
        st.error(
            "⚠️ **OPENAI_API_KEY not set.** "
            "Set it with: `export OPENAI_API_KEY=sk-...`"
        )
    elif not kb_ok:
        st.warning(
            "⚠️ Knowledge base not found — product recommendations won't work.\n\n"
            "Run `python generate_kb_v2.py` first to build it."
        )

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

                    # Persist avatar thumbnail for Phase 2
                    avatar = pil_image.copy()
                    avatar.thumbnail((120, 120), Image.LANCZOS)
                    buf = io.BytesIO()
                    avatar.save(buf, format="PNG")
                    st.session_state.face_image_bytes = buf.getvalue()

                    f = features
                    shape = f["face_shape"]
                    undertone = f["skin_color"]["undertone"]
                    brightness_label = (
                        "fair"
                        if f["skin_color"]["brightness"] > 180
                        else "medium"
                        if f["skin_color"]["brightness"] > 100
                        else "deep"
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

    # ── Face thumbnail + profile pill ────────────────────────────────────
    f = st.session_state.features
    shape = f["face_shape"].capitalize()
    undertone = f["skin_color"]["undertone"].capitalize()
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
        # Show face badge + profile (if accumulated)
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

    # ── Render chat history ──────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render product cards for this assistant message
            if (
                msg["role"] == "assistant"
                and idx in st.session_state.product_results
            ):
                render_product_cards(st.session_state.product_results[idx])

    # ── Chat input ───────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about makeup, skincare, products… ✨")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Forence is thinking…"):
                response_data = get_forence_response(
                    user_input=user_input,
                    history=st.session_state.messages,
                    profile=st.session_state.profile,
                    features=st.session_state.features,
                )

            reply = response_data.get(
                "reply", "Sorry, something went wrong — try again!"
            )
            signals = response_data.get("signals", {})

            st.markdown(reply)

            # Update user profile silently
            updates = signals.get("profile_updates", {})
            if updates:
                st.session_state.profile = update_profile(
                    st.session_state.profile, updates
                )

            # Product retrieval + ranking + display
            products_shown: list[dict] = []

            should_show = signals.get("show_products", False) and not signals.get(
                "needs_clarification", False
            ) and not signals.get("is_off_topic", False)

            if should_show:
                product_query = signals.get("product_query") or user_input
                product_filters = signals.get("product_filters") or {}

                with st.spinner("Finding the best products for you…"):
                    raw = query_products(
                        query=product_query,
                        filters=product_filters,
                        n_results=16,
                    )
                    products_shown = rank_products(raw, top_n=4)

                if products_shown:
                    render_product_cards(products_shown)
                else:
                    st.caption(
                        "*I couldn't find matching products right now — "
                        "try a different query!*"
                    )

        # Persist assistant message + product results
        assistant_msg_idx = len(st.session_state.messages)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        if products_shown:
            st.session_state.product_results[assistant_msg_idx] = products_shown
        st.rerun()

    # ── Reset button ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start over with a new photo", type="secondary"):
        for k in _DEFAULTS:
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()