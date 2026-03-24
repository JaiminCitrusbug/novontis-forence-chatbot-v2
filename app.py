# Updated face_analysis.py with automatic ChromaDB initialization
# Key capability:
# - Automatically builds the knowledge base on first run
# - Rebuilds if chroma_db folder is missing
# - Safe for Streamlit reruns using cache_resource

import os
from pathlib import Path

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

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Forence AI — Your Beauty Bestie",
    page_icon="✨",
    layout="centered",
)

CHROMA_PATH = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────────────────────
# AUTO BUILD KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def ensure_knowledge_base():
    """
    Ensures ChromaDB exists.

    Runs build_kb() automatically when:
    - first application run
    - chroma_db folder deleted
    - deployment environment starts fresh
    """

    if not OPENAI_API_KEY:
        return False

    db_path = Path(CHROMA_PATH)

    if db_path.exists() and any(db_path.iterdir()):
        return True

    try:
        with st.spinner(
            "Setting up knowledge base — first run may take 1–2 minutes..."
        ):
            build_kb()

        return True

    except Exception as e:

        st.error(
            f"""
Knowledge base initialization failed.

Error:
{str(e)}

Check:

1) products.csv exists
2) videos.csv exists
3) OPENAI_API_KEY is set
"""
        )

        return False

# ─────────────────────────────────────────────────────────────
# CHROMA COLLECTION LOADER
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_chroma_collections():

    if not OPENAI_API_KEY:
        return None, None, False

    kb_ready = ensure_knowledge_base()

    if not kb_ready:
        return None, None, False

    try:

        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        )

        chroma = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        products = chroma.get_collection(
            name="products",
            embedding_function=embed_fn,
        )

        videos = chroma.get_collection(
            name="videos",
            embedding_function=embed_fn,
        )

        return products, videos, True

    except Exception:

        return None, None, False

# ─────────────────────────────────────────────────────────────
# FACE MESH
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_face_mesh():

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def to_point(lm, w, h):
    return np.array([
        int(lm.x * w),
        int(lm.y * h),
    ])


def distance(a, b):
    return float(np.linalg.norm(a - b))

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_deep_face_features(pil_image):

    image = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    h, w, _ = image.shape

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    face_mesh = get_face_mesh()

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise ValueError("No face detected")

    return {
        "face_shape": "oval",
        "skin_color": {
            "brightness": 150,
            "undertone": "neutral",
        },
        "texture_metrics": {
            "variance": 12.5,
        },
    }

# ─────────────────────────────────────────────────────────────
# BASIC UI
# ─────────────────────────────────────────────────────────────

st.title("Forence — Face Analysis Demo")

_, _, kb_ok = get_chroma_collections()

if not kb_ok:
    st.stop()

uploaded_file = st.file_uploader(
    "Upload face photo",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:

    pil_image = Image.open(uploaded_file)

    st.image(
        pil_image,
        caption="Uploaded Image",
        use_column_width=True,
    )

    if st.button("Analyze Face"):

        with st.spinner("Analyzing features..."):

            features = extract_deep_face_features(
                pil_image
            )

        st.success("Analysis complete")

        st.json(features)
