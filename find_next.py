import psycopg2
import numpy as np
import argparse
from types import SimpleNamespace

def parse_row(r):
    embedding = np.array(r[4].strip("[]").split(","), dtype=np.float32)
    return SimpleNamespace(id=r[0], title=r[1], date=r[2], link=r[3], embedding=embedding)

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v

DB_USER = "rtvdb"
DB_HOST = "rtvdb.postgres.database.azure.com"
DB_PORT = 5432
DB_NAME = "postgres"

def get_candidates(cur, article_id, limit=50):
    cur.execute(
        """
        WITH base AS (
            SELECT id, embedding, date
            FROM articles
            WHERE id = %s
        )
        SELECT *
        FROM (
            SELECT
                a.id,
                a.title,
                a.date,
                a.link,
                a.embedding::text
            FROM articles a
            JOIN base b ON true
            WHERE a.embedding IS NOT NULL
              AND a.date > b.date
              AND a.id <> b.id
            ORDER BY a.embedding <-> b.embedding
            LIMIT %s
        ) a
        ORDER BY a.date ASC;
        """,
        (article_id, limit)
    )
    return cur.fetchall()

def get_next(previous, candidates):
    W_SEMANTIC  = 0.6
    W_DIRECTION = 0.4
    W_TEMPORAL  = 0.09

    prev_emb = _normalize(np.array(previous.embedding, dtype=np.float32))

    cand_embs = np.array([c.embedding for c in candidates], dtype=np.float32)
    cand_embs_norm = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True).clip(1e-10)
    sem_scores = cand_embs_norm @ prev_emb                          # (n,)
    neighborhood_mean = _normalize(cand_embs_norm.mean(axis=0))
    direction = _normalize(neighborhood_mean - prev_emb)
    dir_scores = cand_embs_norm @ direction                         # (n,)
    n = len(candidates)
    time_scores = np.linspace(1.0, 0.0, n)                         # (n,)
    def minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi - lo > 1e-10 else np.zeros_like(arr)

    scores = (
        W_SEMANTIC  * minmax(sem_scores)  +
        W_DIRECTION * minmax(dir_scores)  +
        W_TEMPORAL  * minmax(time_scores)
    )

    best_idx = int(np.argmax(scores))
    return candidates[best_idx]

def _get_db_connection():
    return psycopg2.connect(
        user=DB_USER,
        password="Bravebutterfly2026",
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find next article for current article.")
    parser.add_argument("article_id", help="Origin article ID")
    parser.add_argument("--limit", type=int, default=50, help="Candidate pool size")
    args = parser.parse_args()
    conn = _get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, date, link, embedding::text FROM articles WHERE id = %s",
        (args.article_id,)
    )
    row = cur.fetchone()

    previous = parse_row(row)
    print(f"Origin:  [{previous.date}] {previous.title}\n")

    candidates = [parse_row(r) for r in get_candidates(cur, args.article_id, args.limit)]
    print(f"Scoring {len(candidates)} candidates...\n")

    result = get_next(previous, candidates)
    print(f"Next:  {result.id}  [{result.date}] {result.title}")
    print(f"         {result.link}")

    cur.close()
    conn.close()