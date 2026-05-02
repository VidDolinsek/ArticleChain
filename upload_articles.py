"""
upload_articles.py
------------------
Uploads articles from a YAML file into the articles table in Azure PostgreSQL.
Assumes the table already exists with columns: id, title, thumbnail, link, date, embedding.

Usage:
    python upload_articles.py --yaml mmc.yaml --password your_password
"""

import argparse
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from read_yaml_data import read_yaml


UPSERT_SQL = """
INSERT INTO articles (id, title, thumbnail, link, date, embedding)
VALUES (%s, %s, %s, %s, %s, NULL)
ON CONFLICT (id) DO UPDATE SET
    title     = EXCLUDED.title,
    thumbnail = EXCLUDED.thumbnail,
    link      = EXCLUDED.link,
    date      = EXCLUDED.date;
"""


def upload(yaml_file: str, password: str, batch_size: int = 500):
    print(f"Reading {yaml_file} ...")
    data = read_yaml(yaml_file)

    rows = [
        (str(mongo_id), title, thumbnail, url, date)
        for mongo_id, title, thumbnail, url, date
        in zip(data['_id'], data['title'], data['thumbnail'], data['url'], data['date'])
        if mongo_id is not None
    ]
    print(f"Loaded {len(rows)} articles.")

    print("Connecting to database ...")
    conn = psycopg2.connect(
        user="rtvdb",
        password=password,
        host="rtvdb.postgres.database.azure.com",
        port=5432,
        database="postgres",
    )
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")

    print(f"Uploading in batches of {batch_size} ...")
    for i in tqdm(range(0, len(rows), batch_size)):
        execute_batch(cur, UPSERT_SQL, rows[i : i + batch_size], page_size=batch_size)
        conn.commit()

    cur.close()
    conn.close()
    print("Done. Embedding column left NULL — run embedding step separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload articles to Azure PostgreSQL (no embeddings).")
    parser.add_argument("--yaml",       required=True,           help="Path to the YAML file")
    parser.add_argument("--password",   required=True,           help="Database password for rtvdb")
    parser.add_argument("--batch-size", type=int, default=500,   help="Rows per batch (default: 500)")
    args = parser.parse_args()

    upload(args.yaml, args.password, args.batch_size)