import argparse
import os

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

load_dotenv()

DB_USER = "rtvdb"
DB_HOST = "rtvdb.postgres.database.azure.com"
DB_PORT = 5432
DB_NAME = "postgres"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms


def _project_out(residuals: np.ndarray, basis_vec: np.ndarray) -> None:
    coeffs = residuals @ basis_vec
    residuals -= np.outer(coeffs, basis_vec)


def _parse_embedding(value) -> np.ndarray:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=np.float32)
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            return np.array([], dtype=np.float32)
        return np.fromstring(text, sep=",", dtype=np.float32)
    raise TypeError(f"Unsupported embedding type: {type(value)}")


def _get_db_connection(password: str):
    return psycopg2.connect(
        user=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )


def _fetch_base_article(conn, article_id: str) -> dict:
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.execute(
        """
        SELECT id, title, date, embedding::text
        FROM articles
        WHERE id = %s;
        """,
        (article_id,),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    article_id, title, date, embedding_text = row
    embedding = _parse_embedding(embedding_text)
    if embedding is None or embedding.size == 0:
        raise ValueError(f"Article {article_id} has no embedding.")
    return {
        "id": article_id,
        "title": title,
        "date": date,
        "embedding": embedding,
    }


def _fetch_candidate_articles(conn, article_id: str, limit: int) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
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
                a.lead,
                a.date,
                a.link,
                a.thumbnail,
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
        (article_id, limit),
    )
    rows = cur.fetchall()
    cur.close()

    candidates = []
    for row in rows:
        cand_id, title, lead, date, link, thumbnail, embedding_text = row
        embedding = _parse_embedding(embedding_text)
        if embedding is None or embedding.size == 0:
            continue
        candidates.append(
            {
                "id": cand_id,
                "title": title,
                "lead": lead,
                "date": date,
                "link": link,
                "thumbnail": thumbnail,
                "embedding": embedding,
            }
        )
    return candidates

def _greedy_volume_max_with_origin(
    origin_norm: np.ndarray,     # shape (d,), already normalized
    vectors_norm: np.ndarray,    # shape (n, d)
    x: int,
    seed_idx: int = 0,
) -> list[int]:
    n, d = vectors_norm.shape
    selected  = [seed_idx]
    remaining = list(range(n))
    remaining.remove(seed_idx)
    residuals = vectors_norm.copy()
    _project_out(residuals, origin_norm)

    new_basis_vec = residuals[seed_idx].copy()
    norm = np.linalg.norm(new_basis_vec)
    if norm > 1e-10:
        new_basis_vec /= norm
        _project_out(residuals, new_basis_vec)

    for _ in range(x - 1):
        if not remaining:
            break
        norms = np.linalg.norm(residuals[remaining], axis=1)
        best_local  = int(np.argmax(norms))
        best_global = remaining[best_local]

        selected.append(best_global)
        remaining.remove(best_global)

        new_basis_vec = residuals[best_global].copy()
        norm = np.linalg.norm(new_basis_vec)
        if norm < 1e-10:
            break
        new_basis_vec /= norm
        _project_out(residuals, new_basis_vec)

    return selected


def give_x_diverse_volume(
    origin: np.ndarray,
    vectors: np.ndarray,
    x: int,
) -> list[int]:
    origin_norm = _normalize(origin.reshape(1, -1))[0]
    vectors_norm = _normalize(vectors)

    relevance = cos_sim(origin_norm.reshape(1, -1), vectors_norm)[0]
    seed = int(np.argmax(relevance))

    return _greedy_volume_max_with_origin(origin_norm, vectors_norm, x, seed_idx=seed)


def give_x_diverse_maxmin(
    origin: np.ndarray,
    vectors: np.ndarray,
    x: int,
) -> list[int]:
    origin_norm = _normalize(origin.reshape(1, -1))
    vectors_norm = _normalize(vectors)

    relevance = cos_sim(origin_norm, vectors_norm)[0]
    seed = int(np.argmax(relevance))
    selected = [seed]
    remaining = list(range(len(vectors_norm)))
    remaining.remove(seed)

    while remaining and len(selected) < x:
        sims = cos_sim(vectors_norm[remaining], vectors_norm[selected])
        max_sims = np.max(sims, axis=1)
        best_local = int(np.argmin(max_sims))
        best_global = remaining[best_local]
        selected.append(best_global)
        remaining.remove(best_global)

    return selected


def give_x_diverse_weighted(
    origin: np.ndarray,
    vectors: np.ndarray,
    x: int,
    relevance_weight: float = 0.3,
) -> list[int]:
    origin_norm = _normalize(origin.reshape(1, -1))[0]
    vectors_norm = _normalize(vectors)

    relevance = cos_sim(origin_norm.reshape(1, -1), vectors_norm)[0]
    rel_min, rel_max = relevance.min(), relevance.max()
    rel_norm = (relevance - rel_min) / (rel_max - rel_min + 1e-10)

    seed = int(np.argmax(relevance))
    selected = [seed]
    remaining = list(range(len(vectors_norm)))
    remaining.remove(seed)
    residuals = vectors_norm.copy()
    _project_out(residuals, origin_norm)

    new_basis_vec = residuals[seed].copy()
    norm = np.linalg.norm(new_basis_vec)
    if norm > 1e-10:
        new_basis_vec /= norm
        _project_out(residuals, new_basis_vec)

    lam = relevance_weight
    for _ in range(x - 1):
        if not remaining:
            break
        res_norms = np.linalg.norm(residuals[remaining], axis=1)
        rn_min, rn_max = res_norms.min(), res_norms.max()
        res_norms_norm = (res_norms - rn_min) / (rn_max - rn_min + 1e-10)

        scores = (1 - lam) * res_norms_norm + lam * rel_norm[remaining]
        best_local = int(np.argmax(scores))
        best_global = remaining[best_local]

        selected.append(best_global)
        remaining.remove(best_global)

        new_basis_vec = residuals[best_global].copy()
        norm = np.linalg.norm(new_basis_vec)
        if norm < 1e-10:
            break
        new_basis_vec /= norm
        _project_out(residuals, new_basis_vec)

    return selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find diverse article groups from DB.")
    parser.add_argument("article_id", help="Origin article ID")
    parser.add_argument("--limit", type=int, default=50, help="Candidate pool size")
    parser.add_argument("--x", type=int, default=7, help="Number of articles to select")
    parser.add_argument(
        "--method",
        choices=["volume", "weighted", "maxmin"],
        default="volume",
        help="Selection method",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.3,
        help="Relevance weight for weighted method",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DB_PASSWORD"),
        help="Database password (defaults to DB_PASSWORD env)",
    )
    args = parser.parse_args()

    if not args.password:
        raise ValueError("Database password is required. Set DB_PASSWORD or pass --password.")

    conn = _get_db_connection(args.password)
    try:
        base = _fetch_base_article(conn, args.article_id)
        if not base:
            raise ValueError(f"Article {args.article_id} not found in database.")

        candidates = _fetch_candidate_articles(conn, args.article_id, args.limit)
    finally:
        conn.close()

    if not candidates:
        raise ValueError("No candidate articles found (check embeddings and date filter).")

    vectors = np.vstack([c["embedding"] for c in candidates])
    x = min(args.x, len(candidates))

    if args.method == "weighted":
        idxs = give_x_diverse_weighted(base["embedding"], vectors, x, relevance_weight=args.weight)
    elif args.method == "maxmin":
        idxs = give_x_diverse_maxmin(base["embedding"], vectors, x)
    else:
        idxs = give_x_diverse_volume(base["embedding"], vectors, x)

    selected = [candidates[i] for i in idxs]

    print("Base article:")
    print(f"  {base['date']} | {base['id']} | {base['title']}")
    print("\nSelected articles:")
    for i, art in enumerate(selected, start=1):
        print(f"  {i:02d}. {art['date']} | {art['id']} | {art['title']}")
