"""
update_lead.py
--------------
Updates articles in the database with lead text from a YAML file.
The lead is stored as plain text in the 'lead' column (type: text).

Usage:
    python update_lead.py --yaml mmc.yaml

The lead is a single text field per article (typically 1-3 sentences).
"""

import argparse
import os
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm
from dotenv import load_dotenv

from read_yaml_data import read_yaml


UPDATE_LEAD_SQL = """
UPDATE articles
SET lead = %s
WHERE id = %s;
"""


def update_lead(yaml_file: str, batch_size: int = 500):
    """
    Load articles from YAML and update their lead in the database.
    
    Args:
        yaml_file: Path to the YAML file
        batch_size: Number of updates per batch (default: 500)
    """
    print(f"Reading {yaml_file} ...")
    data = read_yaml(yaml_file)

    # Prepare rows: (lead_text, article_id)
    rows = []
    for article_id, lead_text in zip(
        data['_id'], 
        data['lead']
    ):
        if article_id is None:
            continue
        
        # Use lead text as-is, or empty string if None
        lead = lead_text if lead_text else ""
        
        rows.append((lead, str(article_id)))
    
    print(f"Loaded {len(rows)} articles with lead text.")
    
    # Show statistics
    lead_chars = [len(l) if l else 0 for l in data['lead'] if l is not None]
    if lead_chars:
        print(f"  Min lead length: {min(lead_chars)} characters")
        print(f"  Max lead length: {max(lead_chars)} characters")
        print(f"  Avg lead length: {sum(lead_chars) / len(lead_chars):.0f} characters")


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

    print(f"Updating lead in batches of {batch_size} ...")
    for i in tqdm(range(0, len(rows), batch_size)):
        execute_batch(cur, UPDATE_LEAD_SQL, rows[i : i + batch_size], page_size=batch_size)
        conn.commit()

    cur.close()
    conn.close()
    print("Done. All lead text updated.")


if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Update articles with lead text from YAML file."
    )
    parser.add_argument("--yaml", required=True, help="Path to the YAML file")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per batch (default: 500)")
    args = parser.parse_args()

    update_lead(args.yaml, args.batch_size)
