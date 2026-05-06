"""
update_text_fields.py
---------------------
Updates articles in the database with lead (text) and paragraphs (JSON array)
from a YAML file. Both fields are written in a single UPDATE per article.

The lead is stored as plain text in the 'lead' column.
Paragraphs are stored as a JSON array of strings in the 'paragraphs' column.

Usage:
    python update_text_fields.py --yaml mmc.yaml
"""

import argparse
import json
import os

import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from tqdm import tqdm

from test_transformerjev.read_yaml_data import read_yaml


UPDATE_SQL = """
UPDATE articles
SET lead = %s, paragraphs = %s
WHERE id = %s;
"""


def update_text_fields(yaml_file: str, batch_size: int = 500):
    print(f"Reading {yaml_file} ...")
    data = read_yaml(yaml_file)

    rows = []
    for article_id, lead_value, paragraphs_value in zip(
        data["_id"], data["lead"], data["paragraphs"]
    ):
        if article_id is None:
            continue

        lead = lead_value if lead_value else ""

        if paragraphs_value is not None and hasattr(paragraphs_value, "__iter__") and not isinstance(paragraphs_value, str):
            paragraphs_json = json.dumps([str(p) for p in paragraphs_value if p is not None])
        else:
            paragraphs_json = json.dumps([])

        rows.append((lead, paragraphs_json, str(article_id)))

    print(f"Loaded {len(rows)} articles.")

    lead_chars = [len(lead) for lead, _, _ in rows if lead]
    para_chars = [len(para) for _, para, _ in rows if para != "[]"]
    if lead_chars:
        print(f"  Lead   — min: {min(lead_chars)}, max: {max(lead_chars)}, avg: {sum(lead_chars) // len(lead_chars)} chars")
    if para_chars:
        print(f"  Paragraphs — min: {min(para_chars)}, max: {max(para_chars)}, avg: {sum(para_chars) // len(para_chars)} chars")

    print("Connecting to database ...")
    conn = psycopg2.connect(
        user="rtvdb",
        password=os.getenv("DB_PASSWORD"),
        host="rtvdb.postgres.database.azure.com",
        port=5432,
        database="postgres",
    )
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")

    print(f"Updating in batches of {batch_size} ...")
    for i in tqdm(range(0, len(rows), batch_size)):
        execute_batch(cur, UPDATE_SQL, rows[i : i + batch_size], page_size=batch_size)
        conn.commit()

    cur.close()
    conn.close()
    print("Done. Lead and paragraphs updated.")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Update articles with lead and paragraphs from a YAML file."
    )
    parser.add_argument("--yaml", required=True, help="Path to the YAML file")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per batch (default: 500)")
    args = parser.parse_args()

    update_text_fields(args.yaml, args.batch_size)
