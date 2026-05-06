"""
find_similarity.py
------------------
Finds semantic similarity between two articles using LLM.
Fetches articles from Azure PostgreSQL database by ID and compares them.

Usage:
    from find_similarity import find_similarity
    result = find_similarity(article_id_1, article_id_2, db_password)
    print(result)  # {"connection": "...", "entities": [...]}
"""

import os
import json
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model_name = "DeepSeek-V4-Flash"
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
db_password = os.getenv("DB_PASSWORD")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)


def fetch_article_from_db(article_id: str) -> dict:
    """
    Fetch article data from PostgreSQL database by ID.
    
    Args:
        article_id: Article ID to fetch
    
    Returns:
        dict with keys: title, lead, paragraphs (or None if not found)
    """
    conn = psycopg2.connect(
        user="rtvdb",
        password=db_password,
        host="rtvdb.postgres.database.azure.com",
        port=5432,
        database="postgres",
    )
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    
    # Fetch article
    cur.execute(
        "SELECT title, lead, paragraphs FROM articles WHERE id = %s",
        (article_id,)
    )
    
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if not result:
        return None
    
    title, lead, paragraphs_json = result
    
    # Parse paragraphs from JSON
    paragraphs = []
    if paragraphs_json:
        try:
            paragraphs = json.loads(paragraphs_json)
        except json.JSONDecodeError:
            paragraphs = []
    
    return {
        "title": title,
        "lead": lead,
        "paragraphs": paragraphs
    }


def find_similarity(article_id_1: str, article_id_2: str) -> dict:
    """
    Find semantic similarity between two articles.
    Fetches articles from database and uses LLM to identify connections.
    
    Args:
        article_id_1: First article ID
        article_id_2: Second article ID
    
    Returns:
        dict with keys:
            - connection: Description of connection (max 15 words)
            - entities: List of common entities
    
    Raises:
        ValueError: If either article is not found
    """
    print(f"Fetching articles {article_id_1} and {article_id_2}...")
    
    article1 = fetch_article_from_db(article_id_1)
    article2 = fetch_article_from_db(article_id_2)
    
    if not article1:
        raise ValueError(f"Article {article_id_1} not found in database")
    if not article2:
        raise ValueError(f"Article {article_id_2} not found in database")
    
    # Build text from paragraphs
    text1 = "\n\n".join(article1["paragraphs"]) if article1["paragraphs"] else ""
    text2 = "\n\n".join(article2["paragraphs"]) if article2["paragraphs"] else ""
    
    title1 = article1["title"]
    lead1 = article1["lead"]
    
    title2 = article2["title"]
    lead2 = article2["lead"]
    
    prompt = f"""
Dobiš dva novinarska članka v slovenščini.

ČLANEK 1:
Naslov: {title1}
Uvod: {lead1}
Besedilo: {text1}

ČLANEK 2:
Naslov: {title2}
Uvod: {lead2}
Besedilo: {text2}

NALOGA:
Vrni SAMO veljavni JSON v tej obliki:
{{
  "connection": "<največ 15 besed>",
  "entities": ["entiteta1", "entiteta2", ...]
}}

Pravila:
- Največ 15 besed za "connection"
- Napiši povedno razlago povezave med članki, ne naštevaj samo entitet
- Osredotoči se na skupne poimenovane entitete ali jasno vsebinsko prekrivanje
- Primer dobrega opisa: "Oba članka poročata o delovanju slovenske vlade in njenih predstavnikov"
- Primer slabega opisa: "Skupne entitete: Golob, vlada"
- Brez dodatnega besedila zunaj JSON
"""
    
    print("Calling LLM...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Output only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        timeout=60.0
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON: {content}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find similarity between two articles")
    parser.add_argument("id1", help="First article ID")
    parser.add_argument("id2", help="Second article ID")
    args = parser.parse_args()
    
    try:
        result = find_similarity(args.id1, args.id2)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")