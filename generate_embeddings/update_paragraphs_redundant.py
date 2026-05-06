"""
update_paragraphs.py
-------------------
Updates articles in the database with paragraphs from a YAML file.
The paragraphs are stored as a JSON array in the 'paragraphs' column (type: text).

Usage:
    python update_paragraphs.py --yaml mmc.yaml --password your_password

Each article can have multiple paragraphs. The paragraphs are stored as:
    - JSON array: ["paragraph 1", "paragraph 2", ...]
    - This format allows for:
      * Easy parsing in Python/JavaScript
      * Individual paragraph access
      * Flexibility for different paragraph counts (typically 5-30 per article)
"""

import argparse
import json
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from test_transformerjev.read_yaml_data import read_yaml


UPDATE_PARAGRAPHS_SQL = """
UPDATE articles
SET paragraphs = %s
WHERE id = %s;
"""


def update_paragraphs(yaml_file: str, password: str, batch_size: int = 500):
    """
    Load articles from YAML and update their paragraphs in the database.
    
    Args:
        yaml_file: Path to the YAML file
        password: Database password for rtvdb
        batch_size: Number of updates per batch (default: 500)
    """
    print(f"Reading {yaml_file} ...")
    data = read_yaml(yaml_file)

    # Prepare rows: (paragraphs_json, article_id)
    rows = []
    for article_id, paragraphs_list in zip(
        data['_id'], 
        data['paragraphs']
    ):
        if article_id is None:
            continue
        
        # Convert paragraphs list to JSON string
        # If paragraphs_list is None or empty, store empty array
        if paragraphs_list:
            paragraphs_json = json.dumps(paragraphs_list, ensure_ascii=False)
        else:
            paragraphs_json = json.dumps([], ensure_ascii=False)
        
        rows.append((paragraphs_json, str(article_id)))
    
    print(f"Loaded {len(rows)} articles with paragraphs.")
    
    # Show statistics
    para_counts = [len(p) if p else 0 for p in data['paragraphs'] if p is not None]
    if para_counts:
        print(f"  Min paragraphs per article: {min(para_counts)}")
        print(f"  Max paragraphs per article: {max(para_counts)}")
        print(f"  Avg paragraphs per article: {sum(para_counts) / len(para_counts):.1f}")

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

    print(f"Updating paragraphs in batches of {batch_size} ...")
    for i in tqdm(range(0, len(rows), batch_size)):
        execute_batch(cur, UPDATE_PARAGRAPHS_SQL, rows[i : i + batch_size], page_size=batch_size)
        conn.commit()

    cur.close()
    conn.close()
    print("Done. All paragraphs updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update articles with paragraphs from YAML file."
    )
    parser.add_argument("--yaml",       required=True,           help="Path to the YAML file")
    parser.add_argument("--password",   required=True,           help="Database password for rtvdb")
    parser.add_argument("--batch-size", type=int, default=500,   help="Rows per batch (default: 500)")
    args = parser.parse_args()

    update_paragraphs(args.yaml, args.password, args.batch_size)
