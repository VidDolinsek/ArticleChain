"""
test_recommender_quality.py
---------------------------
Quality test for the article recommender (recommender.py).

WHAT THIS TEST MEASURES
-----------------------
We do not have human-labelled "correct" articles for each query, so we use a
cheap, automatic proxy for relevance:

    A retrieved article is considered "relevant" if the words of the query
    (or, more strictly, the named entities of the query) appear in the
    article's TITLE or LEAD.

The intuition: if you search for "Tadej Pogacar Tour de France", a good
recommender should return articles that actually mention Pogacar / Tour de
France in their title or lead.

We compute two numbers per query:

    hit_rate     - fraction of retrieved articles that contain AT LEAST ONE
                   query term in title/lead.
    avg_coverage - average fraction of query terms found per article
                   (how completely each article matches the query).

Both are also computed using only the NAMED ENTITIES of the query, which is a
stricter and usually more meaningful signal for news search.

THREE RETRIEVAL STRATEGIES ARE COMPARED
---------------------------------------
1. SEMANTIC ONLY  - rank purely by cosine similarity between the query
                    embedding and the article embeddings (pgvector).
2. WEIGHTED SUM   - normalize the semantic score and the lexical (full-text)
                    score to [0, 1] and combine them as
                        0.5 * semantic + 0.5 * lexical.
3. RRF            - the strategy actually implemented in recommender.py
                    (Reciprocal Rank Fusion of the semantic and lexical
                    rankings). We call search_articles() directly so we test
                    the real production code.

NOTES / LIMITATIONS
-------------------
* Named entities are extracted with CLASSLA-Stanza
  (classla.Pipeline('sl', processors='tokenize,ner')). The pipeline is loaded
  lazily on the first call to extract_entities() and reused for all queries.
* Slovenian is highly inflected ("Pogacar" -> "Pogacarja"). Exact matching
  would miss inflected forms, so we use a crude stemmer (drop the last 1-2
  characters) before substring matching.

USAGE
-----
    cd query_to_recommendation
    python test_recommender_quality.py                # uses built-in queries
    python test_recommender_quality.py "vojna v Ukrajini" "volitve"
    python test_recommender_quality.py --top-k 10
"""

import argparse
import classla

# Reuse the database connection, model loader and the real RRF search.
from recommender import connect_db, load_model, embed_query, search_articles

_nlp_pipeline = None


def _get_nlp_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        classla.download("sl", processors="tokenize,ner", logging_level="ERROR")
        _nlp_pipeline = classla.Pipeline("sl", processors="tokenize,ner", logging_level="ERROR")
    return _nlp_pipeline


# ---------------------------------------------------------------------------
# Default test queries (Slovenian news topics)
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "Tadej Pogacar Tour de France",
    "vojna v Ukrajini",
    "volitve v Sloveniji",
    "poplave na Koroskem",
    "Luka Doncic NBA",
    "cene goriva",
    "Evropska unija sankcije",
    "podnebne spremembe",
]


# ---------------------------------------------------------------------------
# Text helpers - turning a query into terms / entities, and matching them
# ---------------------------------------------------------------------------

# Small Slovenian stopword list: these words carry no topical meaning, so we
# drop them when extracting query terms.
SLOVENE_STOPWORDS = {
    "in", "ali", "pa", "ter", "v", "na", "za", "z", "s", "o", "do", "od",
    "po", "pri", "iz", "k", "h", "je", "so", "bo", "bi", "se", "ne", "da",
    "kot", "ki", "the", "de", "a", "ob", "med", "če", "ce",
}


def extract_query_terms(query: str) -> list[str]:
    """
    Split the query into lowercase content words (stopwords removed).

    Example: "vojna v Ukrajini" -> ["vojna", "ukrajini"]
    """
    words = query.lower().split()
    # keep words that are not stopwords and have at least 2 characters
    return [w for w in words if w not in SLOVENE_STOPWORDS and len(w) > 1]


def extract_entities(query: str) -> list[str]:
    """
    Extract named entities from the query using CLASSLA-Stanza NER.

    Example: "Tadej Pogacar Tour de France"
             -> ["Tadej Pogacar", "Tour de France"]  (lowercased on return)
    """
    nlp = _get_nlp_pipeline()
    doc = nlp(query)
    return [ent.text.lower() for ent in doc.entities]


def crude_stem(word: str) -> str:
    """
    Very crude stemmer for Slovenian: drop the last 1-2 characters so that
    different inflected forms share the same stem.

    Example: "pogacarja" -> "pogacar", "ukrajini" -> "ukrajin"

    Short words (<= 4 chars) are left untouched to avoid over-trimming.
    """
    if len(word) <= 4:
        return word
    if len(word) <= 6:
        return word[:-1]
    return word[:-2]


def term_matches_text(term: str, text: str) -> bool:
    """
    Return True if `term` appears in `text`, allowing for Slovenian
    inflection by comparing crude stems via substring matching.
    """
    text_lower = text.lower()
    term_stem = crude_stem(term.lower())
    # substring match of the stem handles suffix inflection
    # ("pogacar" is a substring of "pogacarja")
    return term_stem in text_lower


def count_term_hits(article: dict, terms: list[str]) -> int:
    """
    Count how many of `terms` appear in the article's title or lead.
    """
    text = (article.get("title", "") or "") + " " + (article.get("lead", "") or "")
    return sum(1 for t in terms if term_matches_text(t, text))


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def evaluate(results: list[dict], terms: list[str]) -> dict:
    """
    Compute the quality metrics for one list of retrieved articles.

    Returns a dict with:
        hit_rate     - fraction of articles with >= 1 matching term
        avg_coverage - average (matched terms / total terms) over all articles
    """
    if not results or not terms:
        return {"hit_rate": 0.0, "avg_coverage": 0.0}

    articles_with_a_hit = 0
    total_coverage = 0.0

    for article in results:
        hits = count_term_hits(article, terms)
        if hits > 0:
            articles_with_a_hit += 1
        total_coverage += hits / len(terms)

    return {
        "hit_rate": articles_with_a_hit / len(results),
        "avg_coverage": total_coverage / len(results),
    }


# ---------------------------------------------------------------------------
# Retrieval strategy 1: SEMANTIC ONLY (pure cosine similarity)
# ---------------------------------------------------------------------------

def retrieve_semantic_only(query_vector, conn, top_k: int) -> list[dict]:
    """
    Rank articles purely by embedding similarity to the query.

    The stored embeddings are L2-normalized, so ordering by L2 distance
    (pgvector '<->') is equivalent to ordering by cosine similarity.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, title, lead
            FROM articles
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %(vec)s::vector
            LIMIT %(k)s
            """,
            {"vec": query_vector, "k": top_k},
        )
        rows = cur.fetchall()
    finally:
        cur.close()

    return [{"id": r[0], "title": r[1] or "", "lead": r[2] or ""} for r in rows]


# ---------------------------------------------------------------------------
# Retrieval strategy 2: WEIGHTED SUM (0.5 * semantic + 0.5 * lexical)
# ---------------------------------------------------------------------------

def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of scores into the [0, 1] range."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-10:        # all values equal -> avoid division by zero
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve_weighted_sum(query, query_vector, conn, top_k: int) -> list[dict]:
    """
    Build the same candidate pool the recommender uses (semantic + lexical
    top-500), then rank by a weighted sum of the normalized scores:

        final = 0.5 * normalized_semantic + 0.5 * normalized_lexical
    """
    cur = conn.cursor()
    try:
        # candidate pool part 1: 500 most semantically similar articles
        cur.execute(
            """
            SELECT id
            FROM articles
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> %(vec)s::vector
            LIMIT 500
            """,
            {"vec": query_vector},
        )
        semantic_ids = [row[0] for row in cur.fetchall()]

        # candidate pool part 2: 500 best full-text (lexical) matches
        cur.execute(
            """
            SELECT id
            FROM articles
            WHERE search_vec @@ plainto_tsquery('simple', %(query)s)
            ORDER BY ts_rank(search_vec, plainto_tsquery('simple', %(query)s)) DESC
            LIMIT 500
            """,
            {"query": query},
        )
        lexical_ids = [row[0] for row in cur.fetchall()]

        candidate_ids = list(set(semantic_ids + lexical_ids))
        if not candidate_ids:
            return []

        # fetch both scores for every candidate
        cur.execute(
            """
            SELECT
                id,
                title,
                lead,
                ts_rank(search_vec, plainto_tsquery('simple', %(query)s)) AS lexical_score,
                1 - (embedding <-> %(vec)s::vector) AS semantic_score
            FROM articles
            WHERE id = ANY(%(ids)s)
            """,
            {"ids": candidate_ids, "query": query, "vec": query_vector},
        )
        rows = cur.fetchall()
    finally:
        cur.close()

    if not rows:
        return []

    articles = []
    for article_id, title, lead, lex_score, sem_score in rows:
        articles.append({
            "id": article_id,
            "title": title or "",
            "lead": lead or "",
            "semantic_score": float(sem_score) if sem_score is not None else 0.0,
            "lexical_score": float(lex_score) if lex_score is not None else 0.0,
        })

    # normalize each score type across the whole candidate pool
    sem_norm = _normalize([a["semantic_score"] for a in articles])
    lex_norm = _normalize([a["lexical_score"] for a in articles])

    # weighted sum: each score contributes half
    for i, article in enumerate(articles):
        article["score"] = 0.5 * sem_norm[i] + 0.5 * lex_norm[i]

    articles.sort(key=lambda a: a["score"], reverse=True)
    return articles[:top_k]


# ---------------------------------------------------------------------------
# Retrieval strategy 3: RRF (the real implemented recommender)
# ---------------------------------------------------------------------------

def retrieve_rrf(query, conn, model, top_k: int) -> list[dict]:
    """Call the real recommender (Reciprocal Rank Fusion) from recommender.py."""
    return search_articles(query, conn, model, top_k=top_k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_metrics(label: str, metrics_terms: dict, metrics_entities: dict) -> None:
    """Pretty-print one strategy's metrics for a single query."""
    print(
        f"    {label:<14} "
        f"| all terms: hit_rate={metrics_terms['hit_rate']:.2f} "
        f"coverage={metrics_terms['avg_coverage']:.2f}  "
        f"|| entities: hit_rate={metrics_entities['hit_rate']:.2f} "
        f"coverage={metrics_entities['avg_coverage']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Quality test for recommender.py")
    parser.add_argument("queries", nargs="*", help="Queries to test (optional)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="How many articles each strategy returns")
    args = parser.parse_args()

    queries = args.queries if args.queries else DEFAULT_QUERIES
    top_k = args.top_k

    print("Loading model and connecting to the database...")
    model = load_model()
    conn = connect_db()

    # We accumulate metrics across all queries to print an overall average.
    # Structure: totals[strategy][metric_type] = list of per-query values
    strategies = ["semantic", "weighted", "rrf"]
    totals = {s: {"term_hit": [], "term_cov": [], "ent_hit": [], "ent_cov": []}
              for s in strategies}

    try:
        for query in queries:
            print("\n" + "=" * 78)
            print(f"QUERY: {query!r}")

            terms = extract_query_terms(query)
            entities = extract_entities(query)
            print(f"  query terms   : {terms}")
            print(f"  named entities: {entities if entities else '(none detected)'}")

            # entities can be empty (e.g. lowercase query) -> fall back to terms
            eval_entities = entities if entities else terms

            # embed the query once and reuse it for strategies 1 and 2
            query_vector = embed_query(query, model)

            # run all three strategies
            results = {
                "semantic": retrieve_semantic_only(query_vector, conn, top_k),
                "weighted": retrieve_weighted_sum(query, query_vector, conn, top_k),
                "rrf": retrieve_rrf(query, conn, model, top_k),
            }

            # evaluate and print
            for strat in strategies:
                res = results[strat]
                m_terms = evaluate(res, terms)
                m_ents = evaluate(res, eval_entities)
                print_metrics(strat, m_terms, m_ents)

                totals[strat]["term_hit"].append(m_terms["hit_rate"])
                totals[strat]["term_cov"].append(m_terms["avg_coverage"])
                totals[strat]["ent_hit"].append(m_ents["hit_rate"])
                totals[strat]["ent_cov"].append(m_ents["avg_coverage"])
    finally:
        conn.close()

    # ---- overall summary across all queries -------------------------------
    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 78)
    print(f"OVERALL AVERAGE over {len(queries)} queries (top-{top_k})")
    print("=" * 78)
    print(f"{'strategy':<14} | {'term hit':>9} {'term cov':>9} "
          f"|| {'ent hit':>9} {'ent cov':>9}")
    print("-" * 78)
    for strat in strategies:
        t = totals[strat]
        print(f"{strat:<14} | {avg(t['term_hit']):>9.3f} {avg(t['term_cov']):>9.3f} "
              f"|| {avg(t['ent_hit']):>9.3f} {avg(t['ent_cov']):>9.3f}")
    print("-" * 78)
    print("Higher is better. 'hit' = fraction of articles with >=1 match, "
          "'cov' = avg fraction of terms matched per article.")


if __name__ == "__main__":
    main()
