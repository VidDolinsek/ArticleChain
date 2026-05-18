"""
recommender_weighted.py
-----------------------
Embed a user query and retrieve the most similar articles from the database.

Ranking strategy: weighted sum of min-max-normalised semantic and lexical scores.

    final = SEMANTIC_WEIGHT * norm(semantic) + LEXICAL_WEIGHT * norm(lexical)

The model must match the one used to generate the stored embeddings:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (dim=384)

Public API:
    connect_db()                         -> psycopg2 connection
    load_model()                         -> SentenceTransformer
    search_articles(query, conn, model)  -> list of article dicts
"""

import argparse
import json
import os

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 20

SEMANTIC_WEIGHT = 0.5
LEXICAL_WEIGHT = 0.5

DB_CONFIG = {
    "user": "rtvdb",
    "password": os.getenv("DB_PASSWORD") or "KiliForPresident2026",
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
    if not query or not query.strip():
        return []

    query = query.strip()
    query_vector = None
    semantic_ok = True
    try:
        query_vector = embed_query(query, model)
    except Exception:
        semantic_ok = False

    semantic_ids = []
    lexical_ids = []

    cur = conn.cursor()
    try:
        if semantic_ok:
            cur.execute(
                """
                SELECT id
                FROM articles
                WHERE embedding IS NOT NULL
                ORDER BY embedding <-> %(vec)s::vector
                LIMIT 500
                """,
                {"vec": query_vector},
            )
            semantic_ids = [row[0] for row in cur.fetchall()]

        cur.execute(
            """
            SELECT id,
                   ts_rank(search_vec, plainto_tsquery('simple', %(query)s)) AS rank
            FROM articles
            WHERE search_vec @@ plainto_tsquery('simple', %(query)s)
            ORDER BY rank DESC
            LIMIT 500
            """,
            {"query": query},
        )
        lexical_ids = [row[0] for row in cur.fetchall()]
    finally:
        cur.close()

    candidate_ids = list(set(semantic_ids + lexical_ids))
    if not candidate_ids:
        return []

    lexical_available = len(lexical_ids) > 0
    cur = conn.cursor()
    try:
        if semantic_ok:
            cur.execute(
                """
                SELECT
                    id,
                    title,
                    lead,
                    date,
                    link,
                    thumbnail,
                    ts_rank(search_vec, plainto_tsquery('simple', %(query)s)) AS lexical_score,
                    1 - (embedding <-> %(vec)s::vector) AS semantic_score
                FROM articles
                WHERE id = ANY(%(ids)s)
                """,
                {"ids": candidate_ids, "query": query, "vec": query_vector},
            )
        else:
            cur.execute(
                """
                SELECT
                    id,
                    title,
                    lead,
                    date,
                    link,
                    thumbnail,
                    ts_rank(search_vec, plainto_tsquery('simple', %(query)s)) AS lexical_score
                FROM articles
                WHERE id = ANY(%(ids)s)
                """,
                {"ids": candidate_ids, "query": query},
            )
        rows = cur.fetchall()
    finally:
        cur.close()

    if not rows:
        return []

    raw_results = []
    for row in rows:
        if semantic_ok:
            article_id, title, lead, date, link, thumbnail, lex_score, sem_score = row
            sem_score = float(sem_score) if sem_score is not None else 0.0
        else:
            article_id, title, lead, date, link, thumbnail, lex_score = row
            sem_score = 0.0

        lex_score = float(lex_score) if lex_score is not None else 0.0

        raw_results.append(
            {
                "id": article_id,
                "title": title or "(brez naslova)",
                "lead": lead or "",
                "date": str(date) if date else "",
                "link": link or "",
                "thumbnail": thumbnail or "",
                "semantic_score": sem_score,
                "lexical_score": lex_score,
            }
        )

    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-10:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    semantic_scores = [r["semantic_score"] for r in raw_results]
    lexical_scores = [r["lexical_score"] for r in raw_results]

    semantic_norm = _normalize(semantic_scores) if semantic_ok else [0.0] * len(raw_results)
    lexical_norm = _normalize(lexical_scores) if lexical_available else [0.0] * len(raw_results)

    for idx, item in enumerate(raw_results):
        if not semantic_ok:
            final_score = lexical_norm[idx]
        elif not lexical_available:
            final_score = semantic_norm[idx]
        else:
            final_score = SEMANTIC_WEIGHT * semantic_norm[idx] + LEXICAL_WEIGHT * lexical_norm[idx]
        item["score"] = round(float(final_score), 4)

    raw_results.sort(key=lambda x: x["score"], reverse=True)

    return [
        {
            "id": item["id"],
            "title": item["title"],
            "lead": item["lead"],
            "date": item["date"],
            "link": item["link"],
            "thumbnail": item["thumbnail"],
            "score": item["score"],
        }
        for item in raw_results[:top_k]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for weighted hybrid search.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--semantic-weight", type=float, default=SEMANTIC_WEIGHT)
    parser.add_argument("--lexical-weight", type=float, default=LEXICAL_WEIGHT)
    args = parser.parse_args()

    SEMANTIC_WEIGHT = args.semantic_weight
    LEXICAL_WEIGHT = args.lexical_weight

    conn = connect_db()
    try:
        model = load_model()
        results = search_articles(args.query, conn, model, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        conn.close()
