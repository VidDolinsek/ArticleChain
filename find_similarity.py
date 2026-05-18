import os
import json
import psycopg2
from dotenv import load_dotenv

load_dotenv()

model_name = "DeepSeek-V4-Flash"
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
db_password = os.getenv("DB_PASSWORD")
_client = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


LLM_SIMILARITY_ENABLED = (
    not _env_flag("DISABLE_LLM_SIMILARITY", False)
    and _env_flag("ENABLE_LLM_SIMILARITY", True)
)


def set_llm_similarity_enabled(enabled: bool) -> None:
    global LLM_SIMILARITY_ENABLED
    LLM_SIMILARITY_ENABLED = enabled


def is_llm_similarity_enabled() -> bool:
    return LLM_SIMILARITY_ENABLED


def _get_client():
    global _client
    if _client is None:
        if not endpoint or not api_key:
            raise ValueError("Missing ENDPOINT or API_KEY for LLM similarity calls")
        from openai import OpenAI

        _client = OpenAI(
            base_url=endpoint,
            api_key=api_key
        )
    return _client


def _disabled_similarity() -> dict:
    return {
        "connection": "LLM povezava je izklopljena za predogled.",
        "entities": [],
    }


def _parse_paragraphs(paragraphs_json):
    if not paragraphs_json:
        return []
    if isinstance(paragraphs_json, list):
        return paragraphs_json
    if isinstance(paragraphs_json, str):
        try:
            return json.loads(paragraphs_json)
        except json.JSONDecodeError:
            return []
    return [str(paragraphs_json)]


def _article_text(article: dict) -> str:
    paragraphs = article.get("paragraphs") or []
    if isinstance(paragraphs, list):
        return "\n\n".join(str(paragraph) for paragraph in paragraphs)
    return str(paragraphs)


def _call_similarity_llm(prompt: str) -> dict:
    if not is_llm_similarity_enabled():
        return _disabled_similarity()

    print("Calling LLM...")
    response = _get_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Output only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON: {content}")


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
    return {
        "title": title,
        "lead": lead,
        "paragraphs": _parse_paragraphs(paragraphs_json)
    }


def find_similarity_from_articles(article1: dict, article2: dict) -> dict:
    if not article1 or not article2:
        raise ValueError("Both articles are required")

    text1 = _article_text(article1)
    text2 = _article_text(article2)

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

    return _call_similarity_llm(prompt)

# LLM klic, ki najde podobnosti med članki in jih opiše kot skupek 
def find_similarity_from_all_articles(articles: list[dict]) -> dict:
    if not articles:
        raise ValueError("At least one article is required")
    if len(articles) < 2:
        raise ValueError("At least two articles are required")

    article_blocks = []
    for idx, article in enumerate(articles, start=1):
        title = article.get("title") or ""
        lead = article.get("lead") or ""
        article_blocks.append(
            f"""ČLANEK {idx}:
Naslov: {title}
Uvod: {lead}"""
        )

    joined_articles = "\n\n".join(article_blocks)

    prompt = f"""
Dobiš več novinarskih člankov v slovenščini.

{joined_articles}

NALOGA:
Vrni SAMO veljavni JSON v tej obliki:
{{
  "connection": "<največ 40 besed>",
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

    return _call_similarity_llm(prompt)


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
