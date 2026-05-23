"""
stem_articles.py
----------------
Lemmatise the `title` and `lead` columns of every article using the classla
Slovenian pipeline (tokenise → POS → lemma) and write the results to the
`title_stem` and `lead_stem` columns.

Usage:
    python stem_articles.py              # process all rows where stem columns are NULL
    python stem_articles.py --limit 100  # process at most 100 rows (for testing)
    python stem_articles.py --all        # re-process every row, overwriting existing stems
"""

import argparse
import os

import classla
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BATCH_SIZE = 64

DB_CONFIG = {
    "user": "rtvdb",
    "password": os.getenv("DB_PASSWORD"),
    "host": "rtvdb.postgres.database.azure.com",
    "port": 5432,
    "database": "postgres",
}

def connect_db():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.close()
    return conn


def ensure_stem_columns(conn):
    cur = conn.cursor()
    for col in ("title_stem", "lead_stem"):
        cur.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'articles' AND column_name = %s
            """,
            (col,),
        )
        if cur.fetchone() is None:
            print(f"Adding column '{col}' to articles table…")
            cur.execute(f"ALTER TABLE articles ADD COLUMN {col} TEXT;")
    conn.commit()
    cur.close()


def fetch_unprocessed_ids(conn, limit=None, reprocess_all=False):
    cur = conn.cursor()
    if reprocess_all:
        query = "SELECT id FROM articles ORDER BY id"
    else:
        query = (
            "SELECT id FROM articles "
            "WHERE title_stem IS NULL OR lead_stem IS NULL "
            "ORDER BY id"
        )
    if limit:
        query += f" LIMIT {limit}"
    cur.execute(query)
    ids = [row[0] for row in cur.fetchall()]
    cur.close()
    return ids


def fetch_articles_batch(conn, ids):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, lead FROM articles WHERE id = ANY(%s)",
        (ids,),
    )
    rows = cur.fetchall()
    cur.close()
    return [{"id": r[0], "title": r[1] or "", "lead": r[2] or ""} for r in rows]


def update_stems_batch(conn, rows):
    """rows: list of (title_stem, lead_stem, article_id)"""
    cur = conn.cursor()
    psycopg2.extras.execute_batch(
        cur,
        "UPDATE articles SET title_stem = %s, lead_stem = %s WHERE id = %s",
        rows,
        page_size=256,
    )
    conn.commit()
    cur.close()

def build_pipeline() -> classla.Pipeline:
    classla.download("sl", verbose=False)
    return classla.Pipeline("sl", processors="tokenize,pos,lemma", verbose=False)


def lemmatise(text: str, nlp: classla.Pipeline) -> str:
    """Return space-joined lemmas for all tokens in *text*."""
    if not text or not text.strip():
        return ""
    doc = nlp(text)
    lemmas = [
        word.lemma
        for sentence in doc.sentences
        for word in sentence.words
        if word.lemma
    ]
    return " ".join(lemmas)


def main():
    parser = argparse.ArgumentParser(description="Lemmatise article titles and leads with classla.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of articles to process.")
    parser.add_argument("--all", dest="reprocess_all", action="store_true", help="Re-process every row, overwriting existing stems.")
    args = parser.parse_args()

    print("Connecting to database…")
    conn = connect_db()
    ensure_stem_columns(conn)

    ids = fetch_unprocessed_ids(conn, limit=args.limit, reprocess_all=args.reprocess_all)
    print(f"Articles to process: {len(ids)}")
    if not ids:
        print("Nothing to do.")
        conn.close()
        return

    print("Loading classla Slovenian pipeline…")
    nlp = build_pipeline()

    batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]

    with tqdm(total=len(ids), unit="article") as pbar:
        for batch_ids in batches:
            articles = fetch_articles_batch(conn, batch_ids)

            rows = []
            for a in articles:
                title_stem = lemmatise(a["title"], nlp)
                lead_stem = lemmatise(a["lead"], nlp)
                rows.append((title_stem, lead_stem, a["id"]))

            update_stems_batch(conn, rows)
            pbar.update(len(articles))

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
