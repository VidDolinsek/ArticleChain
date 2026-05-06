"""
generate_embeddings.py
----------------------
Batch-generate embeddings for all articles in the Azure PostgreSQL DB and store
them in the pgvector `embedding` column.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (dim=384)
Fields used: title + lead + paragraphs concatenated

Usage:
    python generate_embeddings.py              # embed all articles with NULL embedding
    python generate_embeddings.py --limit 100  # process only the first 100 (for testing)
"""

import argparse
import json
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


load_dotenv()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256


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
    """Cast the embedding column to vector(384) if it isn't already."""
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
        "SELECT id, title, lead, paragraphs FROM articles WHERE id = ANY(%s)",
        (ids,)
    )
    rows = cur.fetchall()
    cur.close()

    articles = []
    for article_id, title, lead, paragraphs_raw in rows:
        paragraphs_text = ""
        if paragraphs_raw:
            # paragraphs column may be text (JSON string) or jsonb (already a Python list)
            if isinstance(paragraphs_raw, list):
                paragraphs = paragraphs_raw
            else:
                try:
                    paragraphs = json.loads(paragraphs_raw)
                except (json.JSONDecodeError, TypeError):
                    paragraphs = []
            if isinstance(paragraphs, list):
                paragraphs_text = " ".join(str(p) for p in paragraphs if p)
        articles.append({
            "id": article_id,
            "title": title or "",
            "lead": lead or "",
            "paragraphs": paragraphs_text,
        })
    return articles


def compute_embeddings(model, articles):
    texts = [
        " ".join(part for part in (a["title"], a["lead"], a["paragraphs"]) if part.strip())
        for a in articles
    ]
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


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

            embeddings = compute_embeddings(model, ordered_articles)
            pairs = [(a["id"], emb) for a, emb in zip(ordered_articles, embeddings)] # pairs of id and embedding
            update_embeddings_batch(conn, pairs) # write to DB the embeddings of the batch of articles

            pbar.update(len(ordered_articles)) # update the progress bar with the number of articles embedded

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
