"""
    To run: uvicorn api_server:app --reload
"""


import os
import sys
import json
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query_to_recommendation.recommender_weighted import load_model, connect_db, search_articles
from find_groups import (
    _fetch_base_article,
    _fetch_candidate_articles,
    _parse_embedding,
    give_x_diverse_volume,
)
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
TIMELESS_CANDIDATE_LIMIT = 1
CHAIN_JOBS = {}
CHAIN_JOBS_LOCK = Lock()
BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=4)


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


def _create_chain_job(initial_results):
    job_id = str(uuid.uuid4())
    with CHAIN_JOBS_LOCK:
        CHAIN_JOBS[job_id] = {
            "id": job_id,
            "status": "loading_chains",
            "chains_ready": False,
            "summaries_ready": False,
            "results": initial_results,
            "error": None,
        }
    return job_id


def _update_chain_job(job_id, **updates):
    with CHAIN_JOBS_LOCK:
        job = CHAIN_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)


def _get_chain_job(job_id):
    with CHAIN_JOBS_LOCK:
        job = CHAIN_JOBS.get(job_id)
        return deepcopy(job) if job else None


def _build_article_chain(conn, start_article, length=CHAIN_LENGTH, excluded_ids=None):
    chain = []
    current_article = start_article
    used_ids = set(excluded_ids or [])
    used_ids.add(start_article["id"])

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


def _fetch_timeless_candidate_articles(conn, article_id: str, limit: int, excluded_ids=None) -> list[dict]:
    excluded_ids = list(excluded_ids or [])
    cur = conn.cursor()
    cur.execute("SET timezone = 'Europe/Ljubljana';")
    cur.execute(
        """
        WITH base AS (
            SELECT id, embedding
            FROM articles
            WHERE id = %s
        )
        SELECT
            a.id,
            a.title,
            a.lead,
            a.date,
            a.link,
            a.thumbnail,
            a.embedding::text
        FROM articles a
        JOIN base b ON true
        WHERE a.embedding IS NOT NULL
          AND a.id <> b.id
          AND NOT (a.id = ANY(%s))
        ORDER BY a.embedding <-> b.embedding
        LIMIT %s;
        """,
        (article_id, excluded_ids, limit),
    )
    rows = cur.fetchall()
    cur.close()

    candidates = []
    for row in rows:
        cand_id, title, lead, date, link, thumbnail, embedding_text = row
        embedding = _parse_embedding(embedding_text)
        if embedding is None or embedding.size == 0:
            continue
        candidates.append(
            {
                "id": cand_id,
                "title": title,
                "lead": lead,
                "date": date,
                "link": link,
                "thumbnail": thumbnail,
                "embedding": embedding,
            }
        )
    return candidates


def _build_timeless_article_chain(conn, start_article, length=CHAIN_LENGTH, excluded_ids=None):
    chain = []
    current_article = start_article
    used_ids = set(excluded_ids or [])
    used_ids.add(start_article["id"])

    for _ in range(length):
        candidates = _fetch_timeless_candidate_articles(
            conn,
            current_article["id"],
            TIMELESS_CANDIDATE_LIMIT,
            excluded_ids=used_ids,
        )
        if not candidates:
            break

        next_article = candidates[0]
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


def _preload_single_chain_summary(job_id, base_article, result_article, include_llm):
    if not include_llm:
        result_article["route_summary"] = _disabled_similarity()
        result_article["summary_ready"] = True
        _update_chain_job(
            job_id,
            status="complete",
            summaries_ready=True,
            results=[result_article],
        )
        return

    conn = connect_db()
    try:
        path = _ordered_unique_ids([
            base_article["id"],
            result_article["id"],
            *(chain_article["id"] for chain_article in result_article.get("chain", [])),
        ])
        article_texts = _fetch_article_texts(conn, path)
        result_article["route_summary"] = _summarize_article_path(article_texts, path)
        result_article["summary_ready"] = True

        _update_chain_job(
            job_id,
            status="complete",
            summaries_ready=True,
            results=[result_article],
        )
    except Exception as exc:
        _update_chain_job(
            job_id,
            status="error",
            error=str(exc),
        )
    finally:
        conn.close()


def _build_chain_summaries(conn, paths, include_llm):
    if not include_llm:
        return [_disabled_similarity() for _ in paths]

    all_article_ids = _ordered_unique_ids(
        article_id
        for path in paths
        for article_id in path
    )
    article_texts = _fetch_article_texts(conn, all_article_ids)
    summaries = [None] * len(paths)
    max_workers = min(5, len(paths))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_summarize_article_path, article_texts, path): index
            for index, path in enumerate(paths)
        }
        for future in as_completed(future_map):
            summaries[future_map[future]] = future.result()

    return summaries


def _preload_chains_and_summaries(job_id, base_article, selected_articles, include_llm):
    conn = connect_db()
    full_results = []
    selected_ids = {article["id"] for article in selected_articles}
    globally_used_ids = {base_article["id"], *selected_ids}
    try:
        for index, article in enumerate(selected_articles):
            formatted_article = _format_article(
                article,
                _cosine_similarity(base_article["embedding"], article["embedding"]),
            )
            formatted_article["chain"] = _build_article_chain(
                conn,
                article,
                excluded_ids=globally_used_ids,
            )
            formatted_article["chain_ready"] = True
            formatted_article["summary_ready"] = False
            full_results.append(formatted_article)
            globally_used_ids.update(
                chain_article["id"]
                for chain_article in formatted_article["chain"]
            )

            current_job = _get_chain_job(job_id)
            if current_job:
                current_results = current_job["results"]
                current_results[index] = formatted_article
                _update_chain_job(job_id, results=current_results)

        _update_chain_job(
            job_id,
            status="loading_summaries" if include_llm else "complete",
            chains_ready=True,
            summaries_ready=not include_llm,
            results=full_results,
        )

        if include_llm:
            paths = [
                _ordered_unique_ids([
                    base_article["id"],
                    article["id"],
                    *(chain_article["id"] for chain_article in article.get("chain", [])),
                ])
                for article in full_results
            ]
            summaries = _build_chain_summaries(conn, paths, include_llm)
            for article, summary in zip(full_results, summaries):
                article["route_summary"] = summary
                article["summary_ready"] = True

            _update_chain_job(
                job_id,
                status="complete",
                summaries_ready=True,
                results=full_results,
            )
    except Exception as exc:
        _update_chain_job(
            job_id,
            status="error",
            error=str(exc),
        )
    finally:
        conn.close()

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/api/search")
def search(q: str = Query(..., description="Iskalna poizvedba"), k: int = Query(50, description="Število člankov")):
    started_at = time.perf_counter()
    print(f"[api-search] start q={q!r} k={k}", flush=True)
    conn = connect_db()
    try:
        results = search_articles(q, conn, model, top_k=k)
        elapsed = time.perf_counter() - started_at
        print(f"[api-search] complete count={len(results)} elapsed={elapsed:.2f}s", flush=True)
        return {"results": results}
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(f"[api-search] error after {elapsed:.2f}s: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/api/random")
def random_article():
    conn = connect_db()
    try:
        cur = conn.cursor()
        cur.execute("SET timezone = 'Europe/Ljubljana';")
        cur.execute(
            """
            SELECT id, title, lead, date, link, thumbnail
            FROM articles
            WHERE embedding IS NOT NULL
            ORDER BY random()
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            raise HTTPException(status_code=404, detail="No article found")

        article_id, title, lead, date, link, thumbnail = row
        return {
            "article": {
                "id": article_id,
                "title": title or "(brez naslova)",
                "lead": lead or "",
                "date": str(date) if date else "",
                "link": link or "",
                "thumbnail": thumbnail or "",
                "score": 0,
            }
        }
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

@app.get("/api/chain-jobs/{job_id}")
def chain_job(job_id: str):
    job = _get_chain_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Chain job not found")
    return job

@app.get("/api/diverse")
def diverse_articles(
    article_id: str = Query(..., description="ID izbranega članka"),
    x: int = Query(5, ge=3, le=7, description="Število raznolikih člankov"),
    include_llm: bool = Query(True, description="Ali naj ozadje pripravi LLM razlage"),
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
            formatted_article["chain"] = []
            formatted_article["chain_ready"] = False
            formatted_article["summary_ready"] = False
            results.append(formatted_article)

        job_id = _create_chain_job(results)
        BACKGROUND_EXECUTOR.submit(
            _preload_chains_and_summaries,
            job_id,
            base,
            selected,
            include_llm,
        )

        return {
            "results": results,
            "job_id": job_id,
            "status": "loading_chains",
        }
    finally:
        conn.close()


@app.get("/api/timeless-chain")
def timeless_chain(
    article_id: str = Query(..., description="ID izbranega članka"),
):
    conn = connect_db()
    try:
        base = _fetch_base_article(conn, article_id)
        if not base:
            raise HTTPException(status_code=404, detail="Article not found")

        candidates = _fetch_timeless_candidate_articles(
            conn,
            article_id,
            TIMELESS_CANDIDATE_LIMIT,
            excluded_ids={article_id},
        )
        if not candidates:
            return {"results": [], "status": "complete"}

        first_article = candidates[0]
        result = _format_article(
            first_article,
            _cosine_similarity(base["embedding"], first_article["embedding"]),
        )
        result["chain"] = _build_timeless_article_chain(
            conn,
            first_article,
            excluded_ids={base["id"], first_article["id"]},
        )
        result["chain_ready"] = True
        result["summary_ready"] = True
        result["route_summary"] = _disabled_similarity()

        return {
            "results": [result],
            "status": "complete",
        }
    finally:
        conn.close()
