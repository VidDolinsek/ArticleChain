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
    conn = psycopg2.connect(
        user="rtvdb",
        password=db_password,
        host="rtvdb.postgres.database.azure.com",
        port=5432,
        database="postgres",
    )
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
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


def find_similarity_from_articles(article1: dict, article2: dict) -> dict:
    if not article1 or not article2:
        raise ValueError("Both articles are required")

    paragraphs1 = article1.get("paragraphs") or []
    paragraphs2 = article2.get("paragraphs") or []

    text1 = "\n\n".join(paragraphs1) if isinstance(paragraphs1, list) else str(paragraphs1)
    text2 = "\n\n".join(paragraphs2) if isinstance(paragraphs2, list) else str(paragraphs2)

    title1 = article1.get("title") or ""
    lead1 = article1.get("lead") or ""

    title2 = article2.get("title") or ""
    lead2 = article2.get("lead") or ""

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


def find_similarity(article_id_1: str, article_id_2: str) -> dict:
    print(f"Fetching articles {article_id_1} and {article_id_2}...")
    article1 = fetch_article_from_db(article_id_1)
    article2 = fetch_article_from_db(article_id_2)
    if not article1:
        raise ValueError(f"Article {article_id_1} not found in database")
    if not article2:
        raise ValueError(f"Article {article_id_2} not found in database")
    return find_similarity_from_articles(article1, article2)


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