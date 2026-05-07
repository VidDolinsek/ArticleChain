"""
app.py
------
Streamlit frontend for testing the article recommender.

Run:
    cd query_to_recommendation
    streamlit run app.py

The model and DB connection are cached across Streamlit reruns so they are
not recreated on every interaction.
"""

import streamlit as st

from recommender import connect_db, load_model, search_articles

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MMC Priporočilnik",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources (loaded once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Load the sentence transformer model (runs once, cached for the session)."""
    with st.spinner("Nalagam model..."):
        return load_model()


@st.cache_resource
def get_db():
    """Open a DB connection (runs once, cached for the session)."""
    return connect_db()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("📰 MMC Priporočilnik člankov")
st.caption("Vpiši iskalni poizvedbo in prikaži najboljše ujemajoče se članke iz baze.")

# Search bar
query = st.text_input(
    label="Poizvedba",
    placeholder="npr. Tadej Pogačar Tour de France ...",
    label_visibility="collapsed",
)

col_search, col_slider = st.columns([2, 1])
with col_search:
    search_clicked = st.button("🔍 Poišči", type="primary", use_container_width=True)
with col_slider:
    top_k = st.slider("Število rezultatov", min_value=5, max_value=50, value=20, step=5)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

if search_clicked and query.strip():
    model = get_model()
    conn  = get_db()

    with st.spinner(f"Iščem {top_k} najbolj podobnih člankov..."):
        try:
            results = search_articles(query.strip(), conn, model, top_k=top_k)
        except Exception as e:
            # If the connection was dropped reconnect once and retry
            try:
                conn = connect_db()
                st.cache_resource.clear()   # clear so next call gets fresh connection
                results = search_articles(query.strip(), conn, model, top_k=top_k)
            except Exception:
                st.error(f"Napaka pri iskanju: {e}")
                st.stop()

    if not results:
        st.warning("Ni rezultatov. Poskusi z drugačno poizvedbo.")
        st.stop()

    st.success(f"Najdenih {len(results)} člankov za: **{query}**")
    st.divider()

    # Display each result as a card
    for rank, article in enumerate(results, start=1):
        with st.container():
            left, right = st.columns([1, 4])

            # Thumbnail on the left (if available)
            with left:
                if article["thumbnail"]:
                    st.image(article["thumbnail"], use_container_width=True)
                else:
                    # Placeholder box when no thumbnail
                    st.markdown(
                        f"<div style='background:#f0f0f0;height:80px;border-radius:6px;"
                        f"display:flex;align-items:center;justify-content:center;"
                        f"color:#aaa;font-size:2rem'>📄</div>",
                        unsafe_allow_html=True,
                    )

            # Article info on the right
            with right:
                title_text = f"#{rank} — {article['title']}"

                # Make title a clickable link when a URL is available
                if article["link"]:
                    st.markdown(f"### [{title_text}]({article['link']})")
                else:
                    st.markdown(f"### {title_text}")

                # Date + similarity score badge
                meta_parts = []
                if article["date"]:
                    meta_parts.append(f"📅 {article['date'][:10]}")
                meta_parts.append(f"🎯 Podobnost: `{article['score']:.4f}`")
                st.caption("  ·  ".join(meta_parts))

                # Lead text (first 300 chars to keep cards compact)
                if article["lead"]:
                    lead_preview = article["lead"][:300]
                    if len(article["lead"]) > 300:
                        lead_preview += "…"
                    st.write(lead_preview)

            st.divider()

elif search_clicked and not query.strip():
    st.warning("Prosim vpiši poizvedbo pred iskanjem.")
