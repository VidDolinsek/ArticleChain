"""
Streamlit frontend for querying, finding diverse options, building an article route,
and extracting similarities via LLM.

Run with:
    streamlit run frontend_route_app.py
"""

import os
import sys
import numpy as np
import streamlit as st
from types import SimpleNamespace
from dotenv import load_dotenv

# Add the current directory to sys.path so we can import the local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_to_recommendation.recommender import load_model, connect_db, embed_query, search_articles
from find_groups import give_x_diverse_volume, _parse_embedding
from find_next import get_next
from find_similarity import find_similarity

load_dotenv()

st.set_page_config(page_title="Route Recommender", layout="wide")

@st.cache_resource
def get_model():
    """Load the model used to generate embeddings (cached)."""
    return load_model()

# --- UI Flow ---

st.title("🗺️ Article Route Recommender")

# Initialization of session state variables
if "options" not in st.session_state:
    st.session_state.options = None
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "route_data" not in st.session_state:
    st.session_state.route_data = None

query = st.text_input("Enter your query:", placeholder="e.g. Tadej Pogačar Tour de France")

col1, col2, col3 = st.columns(3)
with col1:
    k_limit = st.slider("Candidates Pool Size (k)", min_value=10, max_value=100, value=50)
with col2:
    x_distinct = st.slider("Distinct Options (x)", min_value=1, max_value=15, value=5)
with col3:
    n_steps = st.slider("Route Steps (n)", min_value=1, max_value=10, value=3)

if st.button("🔍 Search Options", type="primary"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Embedding query and finding distinct options..."):
            model = get_model()
            conn = connect_db()
            
            # Fetch top K candidates using the existing functionality from recommender.py
            results = search_articles(query, conn, model, top_k=k_limit)
            
            if not results:
                st.error("No valid candidates found.")
                conn.close()
                st.stop()
                
            # Recommender.py doesn't return the embeddings directly, so we fetch them to cluster options
            ids = tuple([r["id"] for r in results])
            cur = conn.cursor()
            cur.execute("SELECT id, embedding::text FROM articles WHERE id IN %s", (ids,))
            emb_dict = {row[0]: _parse_embedding(row[1]) for row in cur.fetchall()}
            conn.close()
            
            candidates = []
            for r in results:
                emb = emb_dict.get(r["id"])
                if emb is not None and emb.size > 0:
                    candidates.append(SimpleNamespace(
                        id=r["id"], title=r["title"], date=r["date"], link=r["link"], embedding=emb
                    ))
            
            if not candidates:
                st.error("No valid candidates found.")
            else:
                q_vec = embed_query(query, model)
                q_vec_np = np.array(q_vec, dtype=np.float32)
                cand_vectors = np.vstack([c.embedding for c in candidates])
                
                # Find the x diverse options using volume method
                idxs = give_x_diverse_volume(q_vec_np, cand_vectors, min(x_distinct, len(candidates)))
                options = [candidates[i] for i in idxs]
                
                st.session_state.options = options
                st.session_state.selected_option = None
                st.session_state.route_data = None

if st.session_state.options and st.session_state.selected_option is None:
    st.write("### Choose your starting article:")
    for i, opt in enumerate(st.session_state.options):
        col_text, col_btn = st.columns([5, 1])
        with col_text:
            st.write(f"**{opt.title}**  \n*(📅 {opt.date})*")
        with col_btn:
            if st.button("Select", key=f"sel_{i}"):
                st.session_state.selected_option = opt
                st.session_state.route_data = None
                st.rerun()

if st.session_state.selected_option is not None and st.session_state.route_data is None:
    with st.spinner("Building route and generating LLM similarities..."):
        conn = connect_db()
        
        route = [st.session_state.selected_option]
        current = st.session_state.selected_option
        
        # Building the route sequentially using find_next
        for _ in range(n_steps - 1):
            route_ids = tuple(a.id for a in route)
            cur = conn.cursor()
            cur.execute("""
                WITH base AS (SELECT id, embedding, date FROM articles WHERE id = %s)
                SELECT a.id, a.title, a.date, a.link, a.embedding::text
                FROM articles a JOIN base b ON true
                WHERE a.embedding IS NOT NULL AND a.date > b.date AND a.id NOT IN %s
                ORDER BY a.embedding <-> b.embedding LIMIT %s
            """, (current.id, route_ids, k_limit))
            rows = cur.fetchall()
            cur.close()
            
            cands = []
            for r in rows:
                emb = _parse_embedding(r[4])
                if emb is not None and emb.size > 0:
                    cands.append(SimpleNamespace(id=r[0], title=r[1], date=r[2], link=r[3], embedding=emb))
            
            if not cands:
                break
            current = get_next(current, cands)
            route.append(current)

        # Call find_similarity.py function to get the actual LLM generated links 
        similarities = []
        for i in range(len(route) - 1):
            if route[i].id == route[i+1].id:
                similarities.append({"connection": "Isti članek.", "entities": []})
                continue
            try:
                sim = find_similarity(route[i].id, route[i+1].id)
                similarities.append(sim)
            except Exception as e:
                similarities.append({"connection": f"Error formatting via LLM: {e}", "entities": []})

        conn.close()
        
        st.session_state.route_data = {"route": route, "similarities": similarities}

if st.session_state.route_data is not None:
    st.success("Route generated successfully!")
    route = st.session_state.route_data["route"]
    similarities = st.session_state.route_data["similarities"]
    
    for i, art in enumerate(route):
        st.markdown(f"### {i+1}. {art.title}")
        if art.link:
            st.caption(f"📅 {art.date}  |  🔗 [Preberi članek]({art.link})")
        else:
            st.caption(f"📅 {art.date}")
            
        if i > 0:
            sim = similarities[i-1]
            st.info(
                f"**Povezava s prejšnjim člankom:** {sim.get('connection')}\n\n"
                f"**Skupne entitete:** {', '.join(sim.get('entities', []))}"
            )
        else:
            st.info("**Začetni članek:** Izbran na podlagi vaše iskalne poizvedbe.")
        st.divider()