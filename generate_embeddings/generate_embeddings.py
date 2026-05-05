"""
generate_embeddings.py
----------------------
Batch-generate embeddings for all articles in the Azure PostgreSQL DB and store
them in the pgvector `embedding` column.

Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (dim=768)
Fields used (DB columns only): lead (weight 1.0), paragraphs (weight 0.8)

Usage:
    python generate_embeddings.py              # embed all articles with NULL embedding
    python generate_embeddings.py --limit 100  # process only the first 100 (for testing)
"""

import argparse
import json
import os

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768
BATCH_SIZE = 256
FIELD_WEIGHTS = {"lead": 1.0, "paragraphs": 0.8}


def connect_db():
    conn = psycopg2.connect(
        user="rtvdb",
        password=os.getenv("DB_PASSWORD"),
        host="rtvdb.postgres.database.azure.com",
        port=5432,
        database="postgres",
    )
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.close()
    return conn


def ensure_vector_column(conn):
    """Cast the embedding column to vector(768) if it isn't already."""
    cur = conn.cursor()
    cur.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='articles' AND column_name='embedding';"
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Column 'embedding' not found in table 'articles'.")
    cur.close()
    # pgvector type shows as 'USER-DEFINED'; if it's text/null we need to cast
    if row[0].lower() not in ("user-defined",):
        cur = conn.cursor()
        cur.execute(
            f"ALTER TABLE articles ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) "
            f"USING embedding::vector({EMBEDDING_DIM});"
        )
        conn.commit()
        cur.close()
        print(f"Converted embedding column to vector({EMBEDDING_DIM}).")


def fetch_unembedded_ids(conn, limit=None):
    cur = conn.cursor()
    query = "SELECT id FROM articles WHERE embedding IS NULL ORDER BY id"
    if limit:
        query += f" LIMIT {limit}"
    cur.execute(query)
    ids = [row[0] for row in cur.fetchall()]
    cur.close()
    return ids


def fetch_articles_batch(conn, ids):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, lead, paragraphs FROM articles WHERE id = ANY(%s)",
        (ids,)
    )
    rows = cur.fetchall()
    cur.close()

    articles = []
    for article_id, lead, paragraphs_json in rows:
        paragraphs_text = ""
        if paragraphs_json:
            try:
                paragraphs = json.loads(paragraphs_json)
                if isinstance(paragraphs, list):
                    paragraphs_text = " ".join(str(p) for p in paragraphs if p)
            except (json.JSONDecodeError, TypeError):
                pass
        articles.append({
            "id": article_id,
            "lead": lead or "",
            "paragraphs": paragraphs_text,
        })
    return articles


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def compute_weighted_embeddings(model, articles):
    """
    Compute field-weighted, L2-normalized embeddings for a batch of articles.
    Text exceeding the model's 512-token limit is truncated (not chunked).
    Returns np.ndarray of shape (len(articles), EMBEDDING_DIM).
    """
    n = len(articles)
    final = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    weight_sums = np.zeros((n, 1), dtype=np.float32)

    for field, weight in FIELD_WEIGHTS.items():
        texts = [a[field] for a in articles]
        field_embs = model.encode(texts, show_progress_bar=False).astype(np.float32)
        has_text = np.array(
            [1.0 if t.strip() else 0.0 for t in texts], dtype=np.float32
        )[:, None]
        final += field_embs * weight * has_text
        weight_sums += weight * has_text

    weight_sums[weight_sums == 0] = 1.0 # so that the embedding entry remains 0 if no text was extracted
    final = final / weight_sums # turn the weighted sum to a weighted average (normalization for the weights 0.8 and 1.0)
    # normalize so dot product is equal to cosine similarity
    return normalize_embeddings(final) # L2 normalization of the vector v/||v|| (vector devvided by L2 norm of vector)


def update_embeddings_batch(conn, id_embedding_pairs):
    """Bulk-update the embedding column for a list of (id, embedding_vector) pairs."""
    cur = conn.cursor()
    data = [
        (embedding.tolist(), article_id)
        for article_id, embedding in id_embedding_pairs
    ]
    psycopg2.extras.execute_batch(
        cur,
        "UPDATE articles SET embedding = %s::vector WHERE id = %s",
        data,
        page_size=256,
    )
    conn.commit()
    cur.close()


def main():
    parser = argparse.ArgumentParser(description="Generate and store article embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process (for testing)")
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to database...")
    conn = connect_db()
    ensure_vector_column(conn)

    ids = fetch_unembedded_ids(conn, limit=args.limit)
    print(f"Articles to embed: {len(ids)}")

    if not ids:
        print("All articles already have embeddings.")
        conn.close()
        return

    batches = [ids[i:i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    with tqdm(total=len(ids), unit="article") as pbar:
        for batch_ids in batches:
            # fetch the id, lead and paragraphs for a batch of articles
            articles = fetch_articles_batch(conn, batch_ids)

            # Preserve original ordering so embeddings align with ids
            id_to_article = {a["id"]: a for a in articles} # id of articles in batch
            ordered_articles = [id_to_article[i] for i in batch_ids if i in id_to_article] # we have the leads and the paragraphs

            if not ordered_articles:
                pbar.update(len(batch_ids))
                continue

            embeddings = compute_weighted_embeddings(model, ordered_articles)
            pairs = [(a["id"], emb) for a, emb in zip(ordered_articles, embeddings)] # pairs of id and embedding
            update_embeddings_batch(conn, pairs) # write to DB the embeddings of the batch of articles

            pbar.update(len(ordered_articles)) # update the progress bar with the number of articles embedded

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
