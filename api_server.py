import os
import sys
import json
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query_to_recommendation.recommender import load_model, connect_db, search_articles
from find_groups import _fetch_base_article, _fetch_candidate_articles, give_x_diverse_volume
from find_next import get_next
from find_similarity import find_similarity_from_all_articles

app = FastAPI(title="Article Recommender API")

# Enable CORS so the React frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
CHAIN_LENGTH = 5
CHAIN_CANDIDATE_LIMIT = 100


class RouteExplanationRequest(BaseModel):
    article_ids: list[str]
    include_llm: bool = True


class ChainSummaryRequest(BaseModel):
    paths: list[list[str]]
    include_llm: bool = True


def _cosine_similarity(origin, candidate):
    origin_norm = np.linalg.norm(origin)
    candidate_norm = np.linalg.norm(candidate)
    if origin_norm <= 1e-10 or candidate_norm <= 1e-10:
        return 0.0

    score = float(np.dot(origin, candidate) / (origin_norm * candidate_norm))
    return max(0.0, min(1.0, score))


def _format_article(article, score=None):
    return {
        "id": article["id"],
        "title": article.get("title") or "(brez naslova)",
        "date": str(article.get("date") or ""),
        "link": article.get("link") or "",
        "lead": article.get("lead") or "",
        "thumbnail": article.get("thumbnail") or "",
        "score": round(float(score if score is not None else article.get("score", 0.0)), 4),
    }


def _as_namespace(article):
    return SimpleNamespace(**article)


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


def _fetch_article_texts(conn, article_ids):
    if not article_ids:
        return {}

    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.execute(
        """
        SELECT id, title, lead, paragraphs
        FROM articles
        WHERE id = ANY(%s);
        """,
        (article_ids,),
    )
    rows = cur.fetchall()
    cur.close()

    return {
        article_id: {
            "title": title,
            "lead": lead,
            "paragraphs": _parse_paragraphs(paragraphs_json),
        }
        for article_id, title, lead, paragraphs_json in rows
    }


def _disabled_similarity():
    return {
        "connection": "LLM izklopljen za predogled vizualizacije.",
        "entities": [],
    }


def _ordered_unique_ids(article_ids):
    ordered_ids = []
    seen_ids = set()
    for article_id in article_ids:
        if article_id and article_id not in seen_ids:
            ordered_ids.append(article_id)
            seen_ids.add(article_id)
    return ordered_ids


def _summarize_article_path(article_texts, article_ids):
    route_articles = [
        article_texts[article_id]
        for article_id in article_ids
        if article_id in article_texts
    ]
    if len(route_articles) < 2:
        return {
            "connection": "Premalo besedil za skupno povezavo.",
            "entities": [],
        }

    try:
        return find_similarity_from_all_articles(route_articles)
    except Exception as exc:
        return {
            "connection": f"Napaka pri LLM: {exc}",
            "entities": [],
        }


def _build_article_chain(conn, start_article, length=CHAIN_LENGTH):
    chain = []
    current_article = start_article
    used_ids = {start_article["id"]}

    for _ in range(length):
        candidates = [
            candidate
            for candidate in _fetch_candidate_articles(
                conn,
                current_article["id"],
                CHAIN_CANDIDATE_LIMIT,
            )
            if candidate["id"] not in used_ids
        ]
        if not candidates:
            break

        next_choice = get_next(
            _as_namespace(current_article),
            [_as_namespace(candidate) for candidate in candidates],
        )
        next_article = next(
            candidate for candidate in candidates
            if candidate["id"] == next_choice.id
        )

        chain.append(
            _format_article(
                next_article,
                _cosine_similarity(
                    current_article["embedding"],
                    next_article["embedding"],
                ),
            )
        )
        used_ids.add(next_article["id"])
        current_article = next_article

    return chain

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/api/search")
def search(q: str = Query(..., description="Iskalna poizvedba"), k: int = Query(50, description="Število člankov")):
    conn = connect_db()
    try:
        results = search_articles(q, conn, model, top_k=k)
        return {"results": results}
    finally:
        conn.close()


@app.post("/api/route-explanations")
def route_explanations(payload: RouteExplanationRequest):
    article_ids = _ordered_unique_ids(payload.article_ids)

    if len(article_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two article ids are required")

    if not payload.include_llm:
        return {
            "route_similarity": _disabled_similarity(),
        }

    conn = connect_db()
    try:
        article_texts = _fetch_article_texts(conn, article_ids)
    finally:
        conn.close()

    return {
        "route_similarity": _summarize_article_path(article_texts, article_ids),
    }


@app.post("/api/chain-summaries")
def chain_summaries(payload: ChainSummaryRequest):
    paths = [
        _ordered_unique_ids(path)
        for path in payload.paths
    ]
    paths = [path for path in paths if len(path) >= 2]

    if not paths:
        raise HTTPException(status_code=400, detail="At least one path with two article ids is required")

    if not payload.include_llm:
        return {
            "summaries": [
                {
                    "index": index,
                    "article_ids": path,
                    "route_similarity": _disabled_similarity(),
                }
                for index, path in enumerate(paths)
            ]
        }

    all_article_ids = _ordered_unique_ids(
        article_id
        for path in paths
        for article_id in path
    )
    conn = connect_db()
    try:
        article_texts = _fetch_article_texts(conn, all_article_ids)
    finally:
        conn.close()

    return {
        "summaries": [
            {
                "index": index,
                "article_ids": path,
                "route_similarity": _summarize_article_path(article_texts, path),
            }
            for index, path in enumerate(paths)
        ]
    }

@app.get("/api/diverse")
def diverse_articles(
    article_id: str = Query(..., description="ID izbranega članka"),
    x: int = Query(5, ge=3, le=7, description="Število raznolikih člankov"),
):
    conn = connect_db()
    try:
        base = _fetch_base_article(conn, article_id)
        if not base:
            raise HTTPException(status_code=404, detail="Article not found")
        
        candidates = _fetch_candidate_articles(conn, article_id, 100)
        if not candidates:
            return {"results": []}
            
        vectors = np.vstack([c["embedding"] for c in candidates])

        count = min(x, len(candidates))
        idxs = give_x_diverse_volume(base["embedding"], vectors, count)
        selected = [candidates[i] for i in idxs]
        
        results = []
        for art in selected:
            formatted_article = _format_article(
                art,
                _cosine_similarity(base["embedding"], art["embedding"]),
            )
            formatted_article["chain"] = _build_article_chain(conn, art)
            results.append(formatted_article)

        return {"results": results}
    finally:
        conn.close()
