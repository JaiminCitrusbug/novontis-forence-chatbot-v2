# Simplified generate_kb_v2.py
# This file builds the ChromaDB knowledge base.
# It is automatically called by face_analysis.py when needed.

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


def get_embedding_function():

    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set"
        )

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )


def build_kb():

    print("Initializing ChromaDB...")

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

    print("Loading CSV files...")

    df_products = pd.read_csv(
        "products.csv"
    )

    df_videos = pd.read_csv(
        "videos.csv"
    )

    print("Inserting products...")

    products_col.add(
        ids=df_products["product_id"].astype(str).tolist(),
        documents=df_products["name"].tolist(),
        metadatas=df_products.to_dict("records"),
    )

    print("Inserting videos...")

    videos_col.add(
        ids=df_videos["video_id"].astype(str).tolist(),
        documents=df_videos["title"].tolist(),
        metadatas=df_videos.to_dict("records"),
    )

    print()
    print("Knowledge base ready")
    print(f"Location: {CHROMA_PATH}")


if __name__ == "__main__":

    build_kb()
