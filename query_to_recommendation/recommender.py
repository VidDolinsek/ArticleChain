"""
recommender.py
--------------
Embed a user query and retrieve the most similar articles from the database.

The model must match the one used to generate the stored embeddings:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (dim=384)

Public API:
    connect_db()                         -> psycopg2 connection
    load_model()                         -> SentenceTransformer
    search_articles(query, conn, model)  -> list of article dicts
"""

import os

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Must match generate_embeddings.py — changing this breaks similarity search
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Number of results to return by default
DEFAULT_TOP_K = 20

DB_CONFIG = {
    "user": "rtvdb",
    # DB_PASSWORD env var overrides the fallback
    "password": os.getenv("DB_PASSWORD", "Bravebutterfly2026"),
    "host": "rtvdb.postgres.database.azure.com",
    "port": 5432,
    "database": "postgres",
}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def connect_db():
    """Open and return a connection to the Azure PostgreSQL database."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.close()
    return conn


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model() -> SentenceTransformer:
    """Load the multilingual sentence transformer used to generate embeddings."""
    return SentenceTransformer(MODEL_NAME)


def embed_query(query: str, model: SentenceTransformer) -> list:
    """
    Encode a short user query into a normalized vector.

    Returns a plain Python list so psycopg2 can pass it to pgvector.
    """
    vector: np.ndarray = model.encode(query, normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_articles(
    query: str,
    conn,
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Find the top_k articles most similar to the query.

    pgvector's <=> operator computes cosine distance (0 = identical, 2 = opposite),
    so  score = 1 - cosine_distance  gives a similarity in [-1, 1] where 1 = identical.

    Returns a list of dicts, sorted by descending similarity:
        id, title, lead, date, url, score
    """
    query_vector = embed_query(query, model)

    # <=> is coisne distance 
    sql = """
        SELECT
            id,
            title,
            lead,
            date,
            link,
            thumbnail,
            1 - (embedding <=> %(vec)s::vector) AS score 
        FROM articles
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    cur = conn.cursor()
    cur.execute(sql, {"vec": query_vector, "k": top_k})
    rows = cur.fetchall()
    cur.close()

    results = []
    for article_id, title, lead, date, link, thumbnail, score in rows:
        results.append({
            "id":        article_id,
            "title":     title     or "(brez naslova)",
            "lead":      lead      or "",
            "date":      str(date) if date else "",
            "link":      link      or "",
            "thumbnail": thumbnail or "",
            "score":     round(float(score), 4),
        })

    return results
