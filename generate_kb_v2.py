# generate_kb_v2.py
# Production-safe ChromaDB knowledge base builder

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CHROMA_PATH = "chroma_db"

PRODUCTS_COLLECTION = "products"
VIDEOS_COLLECTION = "videos"

EMBEDDING_MODEL = "text-embedding-3-large"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────────────────────────

def get_embedding_function():

    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set"
        )

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )


# ─────────────────────────────────────────────────────────────
# DATA CLEANING UTILITIES
# ─────────────────────────────────────────────────────────────

def clean_text_column(df, column):

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in CSV"
        )

    print(f"Cleaning column: {column}")

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty rows
    df = df[
        df[column] != ""
    ]

    # Limit length (safety)
    df[column] = df[column].str[:8000]

    print(
        f"Remaining rows after cleaning: {len(df)}"
    )

    if len(df) == 0:
        raise ValueError(
            f"No valid data found in column '{column}'"
        )

    return df


def validate_ids(df, column):

    if column not in df.columns:
        raise ValueError(
            f"ID column '{column}' not found"
        )

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

    df = df[
        df[column] != ""
    ]

    if df[column].duplicated().any():

        duplicates = df[
            df[column].duplicated()
        ][column].tolist()

        raise ValueError(
            f"Duplicate IDs detected: {duplicates[:5]}"
        )

    return df


# ─────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────

def build_kb():

    print()
    print("Initializing ChromaDB...")

    chroma = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    embed_fn = get_embedding_function()

    # ─────────────────────────────────────────────────────────
    # RESET COLLECTIONS
    # ─────────────────────────────────────────────────────────

    try:
        chroma.delete_collection(
            PRODUCTS_COLLECTION
        )
        print("Deleted old products collection")

    except Exception:
        print("No existing products collection")

    try:
        chroma.delete_collection(
            VIDEOS_COLLECTION
        )
        print("Deleted old videos collection")

    except Exception:
        print("No existing videos collection")

    # ─────────────────────────────────────────────────────────
    # CREATE COLLECTIONS
    # ─────────────────────────────────────────────────────────

    products_col = chroma.get_or_create_collection(
        name=PRODUCTS_COLLECTION,
        embedding_function=embed_fn,
    )

    videos_col = chroma.get_or_create_collection(
        name=VIDEOS_COLLECTION,
        embedding_function=embed_fn,
    )

    # ─────────────────────────────────────────────────────────
    # LOAD CSV FILES
    # ─────────────────────────────────────────────────────────

    print()
    print("Loading CSV files...")

    if not os.path.exists("products.csv"):
        raise FileNotFoundError(
            "products.csv not found"
        )

    if not os.path.exists("videos.csv"):
        raise FileNotFoundError(
            "videos.csv not found"
        )

    df_products = pd.read_csv(
        "products.csv"
    )

    df_videos = pd.read_csv(
        "videos.csv"
    )

    print(
        f"Products loaded: {len(df_products)}"
    )

    print(
        f"Videos loaded: {len(df_videos)}"
    )

    # ─────────────────────────────────────────────────────────
    # CLEAN DATA
    # ─────────────────────────────────────────────────────────

    df_products = validate_ids(
        df_products,
        "product_id"
    )

    df_products = clean_text_column(
        df_products,
        "name"
    )

    df_videos = validate_ids(
        df_videos,
        "video_id"
    )

    df_videos = clean_text_column(
        df_videos,
        "title"
    )

    # ─────────────────────────────────────────────────────────
    # INSERT PRODUCTS
    # ─────────────────────────────────────────────────────────

    print()
    print("Inserting products...")

    products_col.add(

        ids=df_products[
            "product_id"
        ].tolist(),

        documents=df_products[
            "name"
        ].tolist(),

        metadatas=df_products.to_dict(
            "records"
        ),
    )

    print(
        f"Products inserted: {len(df_products)}"
    )

    # ─────────────────────────────────────────────────────────
    # INSERT VIDEOS
    # ─────────────────────────────────────────────────────────

    print()
    print("Inserting videos...")

    videos_col.add(

        ids=df_videos[
            "video_id"
        ].tolist(),

        documents=df_videos[
            "title"
        ].tolist(),

        metadatas=df_videos.to_dict(
            "records"
        ),
    )

    print(
        f"Videos inserted: {len(df_videos)}"
    )

    # ─────────────────────────────────────────────────────────

    print()
    print("Knowledge base ready")
    print(
        f"Location: {CHROMA_PATH}"
    )


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    build_kb()