# generate_kb_v2.py
# Hardened production-safe KB builder

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

PRODUCTS_COLLECTION = "products"
VIDEOS_COLLECTION = "videos"

EMBEDDING_MODEL = "text-embedding-3-large"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────

def get_embedding_function():

    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set"
        )

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )


# ─────────────────────────────────────────────
# STRICT TEXT VALIDATION
# ─────────────────────────────────────────────

def clean_and_validate(df, text_column, id_column):

    print(f"Validating {text_column}")

    if text_column not in df.columns:
        raise ValueError(
            f"{text_column} column missing"
        )

    if id_column not in df.columns:
        raise ValueError(
            f"{id_column} column missing"
        )

    df[text_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[id_column] = (
        df[id_column]
        .astype(str)
        .str.strip()
    )

    valid_rows = []
    removed = 0

    for _, row in df.iterrows():

        text = row[text_column]
        rid = row[id_column]

        if not text:
            removed += 1
            continue

        if len(text) < 2:
            removed += 1
            continue

        if text.isspace():
            removed += 1
            continue

        if not rid:
            removed += 1
            continue

        valid_rows.append(row)

    df = pd.DataFrame(valid_rows)

    print(
        f"Valid rows: {len(df)}"
    )

    print(
        f"Removed rows: {removed}"
    )

    if len(df) == 0:
        raise ValueError(
            f"No valid rows found in {text_column}"
        )

    return df


# ─────────────────────────────────────────────
# SAFE BATCH INSERT
# ─────────────────────────────────────────────

def insert_in_batches(
    collection,
    ids,
    documents,
    metadatas,
    batch_size=100,
):

    total = len(ids)

    for i in range(0, total, batch_size):

        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        print(
            f"Inserting batch "
            f"{i} → {i + len(batch_ids)}"
        )

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )


# ─────────────────────────────────────────────

def build_kb():

    print("Initializing ChromaDB")

    chroma = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    embed_fn = get_embedding_function()

    try:
        chroma.delete_collection(
            PRODUCTS_COLLECTION
        )
    except Exception:
        pass

    try:
        chroma.delete_collection(
            VIDEOS_COLLECTION
        )
    except Exception:
        pass

    products_col = chroma.get_or_create_collection(
        name=PRODUCTS_COLLECTION,
        embedding_function=embed_fn,
    )

    videos_col = chroma.get_or_create_collection(
        name=VIDEOS_COLLECTION,
        embedding_function=embed_fn,
    )

    print("Loading CSV files")

    df_products = pd.read_csv(
        "products.csv",
        encoding="utf-8",
    )

    df_videos = pd.read_csv(
        "videos.csv",
        encoding="utf-8",
    )

    print(
        "Products loaded:",
        len(df_products)
    )

    print(
        "Videos loaded:",
        len(df_videos)
    )

    df_products = clean_and_validate(
        df_products,
        text_column="name",
        id_column="product_id",
    )

    df_videos = clean_and_validate(
        df_videos,
        text_column="title",
        id_column="video_id",
    )

    print("Preparing product data")

    product_ids = (
        df_products["product_id"]
        .astype(str)
        .tolist()
    )

    product_docs = (
        df_products["name"]
        .astype(str)
        .tolist()
    )

    product_meta = (
        df_products
        .to_dict("records")
    )

    print("Preparing video data")

    video_ids = (
        df_videos["video_id"]
        .astype(str)
        .tolist()
    )

    video_docs = (
        df_videos["title"]
        .astype(str)
        .tolist()
    )

    video_meta = (
        df_videos
        .to_dict("records")
    )

    print("Inserting products")

    insert_in_batches(
        products_col,
        product_ids,
        product_docs,
        product_meta,
    )

    print("Inserting videos")

    insert_in_batches(
        videos_col,
        video_ids,
        video_docs,
        video_meta,
    )

    print("Knowledge base ready")

    print(
        "Location:",
        CHROMA_PATH
    )


if __name__ == "__main__":

    build_kb()