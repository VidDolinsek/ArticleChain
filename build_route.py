import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from dotenv import load_dotenv

from find_groups import (
    give_x_diverse_maxmin,
    give_x_diverse_volume,
    give_x_diverse_weighted,
    _parse_embedding,
)
from find_next import get_next
from find_similarity import find_similarity_from_articles

load_dotenv()

DB_USER = "rtvdb"
DB_HOST = "rtvdb.postgres.database.azure.com"
DB_PORT = 5432
DB_NAME = "postgres"


@dataclass
class ArticleEmbedding:
    id: str
    title: str
    date: Any
    link: str
    embedding: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "date": _serialize_date(self.date),
            "link": self.link,
        }


class ArticleCache:
    def __init__(self) -> None:
        self.meta: Dict[str, ArticleEmbedding] = {}
        self.text: Dict[str, Dict[str, Any]] = {}

    def store_meta(self, article: ArticleEmbedding) -> None:
        self.meta[article.id] = article

    def get_meta(self, article_id: str) -> Optional[ArticleEmbedding]:
        return self.meta.get(article_id)

    def get_text(self, conn, article_id: str) -> Dict[str, Any]:
        cached = self.text.get(article_id)
        if cached:
            return cached

        article = _fetch_article_text(conn, article_id)
        if not article:
            raise ValueError(f"Article {article_id} not found in database")

        self.text[article_id] = article
        return article


def _serialize_date(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _get_db_connection(password: str):
    return psycopg2.connect(
        user=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )


def _fetch_article_text(conn, article_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.execute(
        "SELECT title, lead, paragraphs FROM articles WHERE id = %s",
        (article_id,),
    )
    row = cur.fetchone()
    cur.close()

    if not row:
        return None

    title, lead, paragraphs_json = row
    paragraphs = []
    if paragraphs_json:
        try:
            paragraphs = json.loads(paragraphs_json)
        except json.JSONDecodeError:
            paragraphs = []

    return {
        "title": title,
        "lead": lead,
        "paragraphs": paragraphs,
    }


def fetch_article_meta(conn, article_id: str, cache: ArticleCache) -> Optional[ArticleEmbedding]:
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.execute(
        "SELECT id, title, date, link, embedding::text FROM articles WHERE id = %s",
        (article_id,),
    )
    row = cur.fetchone()
    cur.close()

    if not row:
        return None

    article_id, title, date, link, embedding_text = row
    embedding = _parse_embedding(embedding_text)
    if embedding is None or embedding.size == 0:
        raise ValueError(f"Article {article_id} has no embedding.")

    article = ArticleEmbedding(
        id=article_id,
        title=title,
        date=date,
        link=link,
        embedding=embedding,
    )
    cache.store_meta(article)
    return article


def fetch_candidate_articles(
    conn,
    article_id: str,
    limit: int,
    cache: ArticleCache,
) -> List[ArticleEmbedding]:
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.execute(
        """
        WITH base AS (
            SELECT id, embedding, date
            FROM articles
            WHERE id = %s
        )
        SELECT *
        FROM (
            SELECT
                a.id,
                a.title,
                a.date,
                a.link,
                a.embedding::text
            FROM articles a
            JOIN base b ON true
            WHERE a.embedding IS NOT NULL
              AND a.date > b.date
              AND a.id <> b.id
            ORDER BY a.embedding <-> b.embedding
            LIMIT %s
        ) a
        ORDER BY a.date ASC;
        """,
        (article_id, limit),
    )
    rows = cur.fetchall()
    cur.close()

    candidates: List[ArticleEmbedding] = []
    for row in rows:
        cand_id, title, date, link, embedding_text = row
        embedding = _parse_embedding(embedding_text)
        if embedding is None or embedding.size == 0:
            continue
        article = ArticleEmbedding(
            id=cand_id,
            title=title,
            date=date,
            link=link,
            embedding=embedding,
        )
        cache.store_meta(article)
        candidates.append(article)

    return candidates


def select_diverse_options(
    origin: ArticleEmbedding,
    candidates: List[ArticleEmbedding],
    x: int,
    method: str,
    weight: float,
) -> List[ArticleEmbedding]:
    if not candidates:
        return []

    vectors = np.vstack([c.embedding for c in candidates])
    x = min(x, len(candidates))

    if method == "weighted":
        idxs = give_x_diverse_weighted(origin.embedding, vectors, x, relevance_weight=weight)
    elif method == "maxmin":
        idxs = give_x_diverse_maxmin(origin.embedding, vectors, x)
    else:
        idxs = give_x_diverse_volume(origin.embedding, vectors, x)

    return [candidates[i] for i in idxs]


def prompt_for_choice(options: List[ArticleEmbedding]) -> int:
    while True:
        raw = input(f"Select option (1-{len(options)}): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= choice <= len(options):
            return choice
        print("Choice out of range.")


def build_route(
    conn,
    cache: ArticleCache,
    origin: ArticleEmbedding,
    first_choice: ArticleEmbedding,
    steps: int,
    limit: int,
) -> List[ArticleEmbedding]:
    route = [origin, first_choice]
    current = first_choice

    for _ in range(steps - 1):
        candidates = fetch_candidate_articles(conn, current.id, limit, cache)
        if not candidates:
            break
        current = get_next(current, candidates)
        route.append(current)

    return route


def build_similarities(
    conn,
    cache: ArticleCache,
    route: List[ArticleEmbedding],
) -> List[Dict[str, Any]]:
    pairs = list(zip(route, route[1:]))
    if not pairs:
        return []

    for prev, nxt in pairs:
        cache.get_text(conn, prev.id)
        cache.get_text(conn, nxt.id)

    def _run_similarity(prev_id: str, next_id: str) -> Dict[str, Any]:
        article1 = cache.text[prev_id]
        article2 = cache.text[next_id]
        similarity = find_similarity_from_articles(article1, article2)
        return {
            "from": prev_id,
            "to": next_id,
            "connection": similarity.get("connection"),
            "entities": similarity.get("entities", []),
        }

    results: List[Optional[Dict[str, Any]]] = [None] * len(pairs)
    max_workers = min(8, len(pairs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_similarity, prev.id, nxt.id): idx
            for idx, (prev, nxt) in enumerate(pairs)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()

    return [item for item in results if item is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an article route with explanations.")
    parser.add_argument("article_id", help="Origin article ID")
    parser.add_argument("--x", type=int, default=7, help="Number of initial options")
    parser.add_argument("--limit", type=int, default=50, help="Candidate pool size")
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of hops after origin (includes the chosen option)",
    )
    parser.add_argument(
        "--method",
        choices=["volume", "weighted", "maxmin"],
        default="volume",
        help="Selection method for initial options",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.1,
        help="Relevance weight for weighted method",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DB_PASSWORD"),
        help="Database password (defaults to DB_PASSWORD env)",
    )
    parser.add_argument(
        "--out",
        default="route.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if not args.password:
        raise ValueError("Database password is required. Set DB_PASSWORD or pass --password.")

    cache = ArticleCache()
    conn = _get_db_connection(args.password)
    try:
        origin = fetch_article_meta(conn, args.article_id, cache)
        if not origin:
            raise ValueError(f"Article {args.article_id} not found in database.")

        candidates = fetch_candidate_articles(conn, args.article_id, args.limit, cache)
        if not candidates:
            raise ValueError("No candidate articles found (check embeddings and date filter).")

        options = select_diverse_options(origin, candidates, args.x, args.method, args.weight)
        if not options:
            raise ValueError("No options selected (check inputs).")

        print("Base article:")
        print(f"  {origin.date} | {origin.id} | {origin.title}")
        print("\nOptions:")
        for i, art in enumerate(options, start=1):
            print(f"  {i:02d}. {art.date} | {art.id} | {art.title}")

        choice = prompt_for_choice(options)
        selected = options[choice - 1]

        route = build_route(conn, cache, origin, selected, args.steps, args.limit)
        similarities = build_similarities(conn, cache, route)

        output = {
            "params": {
                "limit": args.limit,
                "x": args.x,
                "steps_requested": args.steps,
                "steps_completed": max(0, len(route) - 1),
                "method": args.method,
                "weight": args.weight if args.method == "weighted" else None,
            },
            "origin": origin.to_dict(),
            "options": [a.to_dict() for a in options],
            "selected_option_index": choice,
            "selected_option": selected.to_dict(),
            "route": [a.to_dict() for a in route],
            "similarities": similarities,
        }

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nSaved route to {args.out}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
