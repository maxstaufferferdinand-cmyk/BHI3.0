from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "Beyond Human Intuition 3.1"
APP_SUBTITLE = "Surgical Technology Hypothesis Generator"

PROJECT_DIR = Path(__file__).resolve().parent
MERGED_CSV_DIR = PROJECT_DIR / "streamlit_prototype_exports" / "merged_cluster_domains"
MERGED_CSV_GLOB = "streamlit_hypotheses_with_merged_domains_*.csv"
FALLBACK_CSV_DIR = PROJECT_DIR / "streamlit_prototype_exports"
FALLBACK_CSV_GLOB = "streamlit_hypotheses_prototype_app_table_*.csv"
ASSET_DIR = PROJECT_DIR / "assets"

SURGICAL_MAP_PATH = ASSET_DIR / "surgical_pair_umap_concept_map.png"
OPENALEX_MAP_PATH = ASSET_DIR / "figure_openalex_engineering_concept_map_adjusted_labels.png"
ROC_PATH = ASSET_DIR / "cluster_edge_strength_ablation_roc.png"

st.set_page_config(
    page_title="BHI3.1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2.4rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}
h1 { letter-spacing: -0.035em; }
h2, h3 { letter-spacing: -0.02em; }
.bhi-small-muted { color: #9aa0a6; font-size: 0.92rem; }
.bhi-caution {
    background: rgba(255, 193, 7, 0.12);
    border-left: 4px solid rgba(255, 193, 7, 0.85);
    padding: 0.9rem 1rem;
    border-radius: 0.4rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.bhi-card {
    border: 1px solid rgba(128, 128, 128, 0.30);
    border-radius: 0.9rem;
    padding: 1.15rem 1.25rem;
    margin-top: 1rem;
    background: rgba(128, 128, 128, 0.06);
}
.bhi-hypothesis-title {
    font-size: 1.42rem;
    font-weight: 750;
    margin-bottom: 0.95rem;
}
.bhi-label {
    color: #9aa0a6;
    font-size: 0.82rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    margin-top: 0.35rem;
    margin-bottom: 0.2rem;
}
.bhi-main-text {
    font-size: 1.05rem;
    line-height: 1.58;
    margin-bottom: 0.85rem;
}
.bhi-meta-box {
    border-top: 1px solid rgba(128, 128, 128, 0.28);
    margin-top: 1.05rem;
    padding-top: 0.9rem;
}
.bhi-meta { font-size: 0.92rem; color: #b8b8b8; }
.bhi-caption {
    color: #9aa0a6;
    font-size: 0.88rem;
    line-height: 1.35;
    margin-top: 0.35rem;
    margin-bottom: 1rem;
}
.bhi-footer-note { color: #9aa0a6; font-size: 0.9rem; }
div.stButton > button[kind="primary"] { font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


@st.cache_data(show_spinner=False)
def load_hypotheses() -> tuple[pd.DataFrame, str]:
    merged_path = _latest_file(MERGED_CSV_DIR, MERGED_CSV_GLOB)

    if merged_path is not None:
        df = _read_csv_safely(merged_path)
        source = str(merged_path)
    else:
        fallback_path = _latest_file(FALLBACK_CSV_DIR, FALLBACK_CSV_GLOB)
        if fallback_path is None:
            raise FileNotFoundError(
                "No hypothesis CSV found. Expected either:\n"
                f"  {MERGED_CSV_DIR / MERGED_CSV_GLOB}\n"
                "or fallback:\n"
                f"  {FALLBACK_CSV_DIR / FALLBACK_CSV_GLOB}"
            )
        df = _read_csv_safely(fallback_path)
        source = str(fallback_path)

    required_cols = [
        "final_rank",
        "hypothesis_title",
        "hypothesis_text",
        "mechanistic_rationale",
        "concept_text",
        "problem_cluster_label",
        "solution_cluster_label",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Hypothesis CSV is missing required columns: {missing}")

    if "merged_surgical_domain" not in df.columns:
        if "surgical_domain_auto" in df.columns:
            df["merged_surgical_domain"] = df["surgical_domain_auto"]
        else:
            df["merged_surgical_domain"] = "All surgical domains"

    if "crosscut_topics" not in df.columns:
        df["crosscut_topics"] = ""

    text_cols = [
        "hypothesis_title",
        "hypothesis_text",
        "mechanistic_rationale",
        "testing_route",
        "main_risk_or_caveat",
        "concept_text",
        "problem_cluster_label",
        "solution_cluster_label",
        "merged_surgical_domain",
        "crosscut_topics",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["final_rank"] = pd.to_numeric(df["final_rank"], errors="coerce")
    df = df.sort_values("final_rank", na_position="last").reset_index(drop=True)

    if "ml_probability_top20_like" in df.columns:
        df["ml_probability_top20_like"] = pd.to_numeric(
            df["ml_probability_top20_like"], errors="coerce"
        )

    return df, source


def format_probability(x: object) -> str:
    try:
        if pd.isna(x):
            return "not available"
        return f"{float(x):.4f}"
    except Exception:
        return "not available"


def init_session_state() -> None:
    if "shown_final_ranks" not in st.session_state:
        st.session_state.shown_final_ranks = []
    if "current_hypothesis_rank" not in st.session_state:
        st.session_state.current_hypothesis_rank = None


def get_unshown_rows(df: pd.DataFrame) -> pd.DataFrame:
    shown = set(st.session_state.get("shown_final_ranks", []))
    return df[~df["final_rank"].astype(str).isin(shown)].copy()


def choose_random_hypothesis(filtered_df: pd.DataFrame) -> pd.Series | None:
    available = get_unshown_rows(filtered_df)
    if len(available) == 0:
        return None

    random_idx = random.choice(list(available.index))
    row = available.loc[random_idx]

    rank_key = str(row.get("final_rank", row.name))
    if rank_key not in st.session_state.shown_final_ranks:
        st.session_state.shown_final_ranks.append(rank_key)

    st.session_state.current_hypothesis_rank = rank_key
    return row


def current_row(df: pd.DataFrame) -> pd.Series | None:
    rank = st.session_state.get("current_hypothesis_rank")
    if rank is None:
        return None
    match = df[df["final_rank"].astype(str).eq(str(rank))]
    if len(match) == 0:
        return None
    return match.iloc[0]


def render_header() -> None:
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.markdown(
        "**BHI3.1** is a cross-domain literature-mining framework designed to identify "
        "novel bridge hypotheses between visceral surgery and engineering concepts."
    )
    st.divider()


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## BHI3.1")
        st.caption("Beyond Human Intuition 3.1")
        st.markdown("### Navigation")

        page = st.radio(
            label="",
            options=["Hypothesis generator", "Concept graphs", "About BHI3.1"],
            label_visibility="collapsed",
        )

        st.divider()
        st.info(
            "Prototype interface only. Hypotheses are pre-generated research candidates; "
            "no live machine-learning or LLM model is called by this webapp."
        )

    return page


def render_project_metrics(df: pd.DataFrame) -> None:
    row1 = st.columns(5)
    row1[0].metric("Surgical abstracts", "323,796")
    row1[1].metric("OpenAlex abstracts", "557,154")
    row1[2].metric("Naive candidate space", "575.6M")
    row1[3].metric("Framework-supported candidates", "11,574,071")
    row1[4].metric("Final selection", "29,500")

    with st.expander("Loaded prototype dataset details", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loaded hypotheses", f"{len(df):,}")
        c2.metric("Loaded surgical domains", f"{df['merged_surgical_domain'].nunique():,}")
        c3.metric("Loaded problem clusters", f"{df['problem_cluster_label'].nunique():,}")
        if "concept_id" in df.columns:
            c4.metric("Loaded OpenAlex concepts", f"{df['concept_id'].nunique():,}")
        else:
            c4.metric("Loaded OpenAlex concepts", f"{df['concept_text'].nunique():,}")

        st.markdown(
            "<div class='bhi-small-muted'>The public prototype uses a reduced, "
            "webapp-ready CSV table derived from pre-generated LLM hypotheses.</div>",
            unsafe_allow_html=True,
        )


def render_hypothesis_card(row: pd.Series) -> None:
    title = row.get("hypothesis_title", "") or "Generated surgical technology hypothesis"

    st.markdown("<div class='bhi-card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='bhi-hypothesis-title'>{title}</div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("<div class='bhi-label'>Surgical domain</div>", unsafe_allow_html=True)
        st.write(row.get("merged_surgical_domain", ""))
    with m2:
        st.markdown("<div class='bhi-label'>OpenAlex / engineering concept</div>", unsafe_allow_html=True)
        st.write(row.get("concept_text", ""))

    st.markdown("<div class='bhi-label'>Hypothesis</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bhi-main-text'>{row.get('hypothesis_text', '')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='bhi-label'>Mechanistic rationale</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bhi-main-text'>{row.get('mechanistic_rationale', '')}</div>", unsafe_allow_html=True)

    optional_cols = st.columns(2)
    with optional_cols[0]:
        if row.get("testing_route", ""):
            st.markdown("<div class='bhi-label'>Testing route</div>", unsafe_allow_html=True)
            st.write(row.get("testing_route", ""))
    with optional_cols[1]:
        if row.get("main_risk_or_caveat", ""):
            st.markdown("<div class='bhi-label'>Main risk / caveat</div>", unsafe_allow_html=True)
            st.write(row.get("main_risk_or_caveat", ""))

    st.markdown("<div class='bhi-meta-box'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='bhi-label'>Surgical problem cluster</div>", unsafe_allow_html=True)
        st.write(row.get("problem_cluster_label", ""))
    with c2:
        st.markdown("<div class='bhi-label'>Historical surgical solution analogy</div>", unsafe_allow_html=True)
        st.write(row.get("solution_cluster_label", ""))

    meta = []
    try:
        meta.append(f"Final rank: {int(row.get('final_rank')):,}")
    except Exception:
        meta.append(f"Final rank: {row.get('final_rank')}")

    if "ml_probability_top20_like" in row.index:
        meta.append(f"ML support: {format_probability(row.get('ml_probability_top20_like'))}")
    if row.get("crosscut_topics", ""):
        meta.append(f"Cross-cutting topics: {row.get('crosscut_topics')}")

    st.markdown(f"<div class='bhi-meta'>{' · '.join(meta)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_hypothesis_generator(df: pd.DataFrame, source: str) -> None:
    render_header()
    render_project_metrics(df)

    st.markdown("## Hypothesis generator")
    st.write(
        "Select a broad surgical field. The generator then randomly displays one "
        "pre-generated hypothesis from all problem clusters within that field."
    )

    st.markdown(
        "<div class='bhi-caution'>"
        "<b>Important:</b> These hypotheses are not clinical recommendations. "
        "They are research candidates generated from literature-mining, graph projection, "
        "ML prioritization, and LLM formulation. They require expert review for novelty, "
        "feasibility, safety, ethics, and biological plausibility."
        "</div>",
        unsafe_allow_html=True,
    )

    domain_counts = df.groupby("merged_surgical_domain").size().sort_values(ascending=False).to_dict()
    domain_options = ["All surgical domains"] + list(domain_counts.keys())

    selected_domain = st.selectbox(
        "Surgical field",
        domain_options,
        format_func=lambda x: x if x == "All surgical domains" else f"{x} ({domain_counts.get(x, 0):,})",
    )

    if selected_domain == "All surgical domains":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["merged_surgical_domain"].eq(selected_domain)].copy()

    available_now = len(get_unshown_rows(filtered_df))

    c1, c2, c3 = st.columns(3)
    c1.metric("Available for selected field", f"{available_now:,}")
    c2.metric("Already shown this session", f"{len(st.session_state.shown_final_ranks):,}")
    c3.metric("Problem clusters in selected field", f"{filtered_df['problem_cluster_label'].nunique():,}")

    b1, b2 = st.columns([1, 1])
    with b1:
        generate = st.button("Generate hypothesis", type="primary", use_container_width=True)
    with b2:
        reset = st.button("Reset shown hypotheses", use_container_width=True)

    if reset:
        st.session_state.shown_final_ranks = []
        st.session_state.current_hypothesis_rank = None
        st.rerun()

    if generate:
        row = choose_random_hypothesis(filtered_df)
        if row is None:
            st.warning(
                "All hypotheses for this surgical field have already been shown in this browser session. "
                "Reset shown hypotheses or select a broader field."
            )
        else:
            render_hypothesis_card(row)
    else:
        row = current_row(df)
        if row is not None:
            st.markdown("### Current hypothesis")
            render_hypothesis_card(row)
        else:
            st.info("Click **Generate hypothesis** to display the first candidate.")

    with st.expander("Prototype source file", expanded=False):
        st.code(source)
        st.markdown(
            "The app uses the newest merged-domain CSV from "
            "`streamlit_prototype_exports/merged_cluster_domains/` when available."
        )


def render_concept_graphs() -> None:
    render_header()

    st.markdown("## Concept graphs and model validation")
    st.write(
        "This section summarizes the surgical concept space, the external OpenAlex "
        "engineering mechanism space, and the model-validation ROC curve."
    )

    st.markdown("### Concept maps")

    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.markdown("#### Surgical problem–solution concept map")
        if SURGICAL_MAP_PATH.exists():
            st.image(str(SURGICAL_MAP_PATH), use_container_width=True)
        else:
            st.warning(f"Image not found: {SURGICAL_MAP_PATH}")
        st.markdown(
            "<div class='bhi-caption'>Two-dimensional UMAP projection of embedded "
            "surgical problem–solution pairs. Background density indicates local "
            "concentration of concepts; red markers label high-frequency historical "
            "problem–solution relations.</div>",
            unsafe_allow_html=True,
        )

    with map_col2:
        st.markdown("#### OpenAlex engineering mechanism concept map")
        if OPENALEX_MAP_PATH.exists():
            st.image(str(OPENALEX_MAP_PATH), use_container_width=True)
        else:
            st.warning(f"Image not found: {OPENALEX_MAP_PATH}")
        st.markdown(
            "<div class='bhi-caption'>Two-dimensional UMAP projection of OpenAlex "
            "engineering mechanism embeddings. Background density indicates local "
            "concentration of concepts; colored points indicate embedding clusters; "
            "representative concepts are labeled.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Surgical edge-strength model")

    left, center, right = st.columns([0.18, 0.64, 0.18])
    with center:
        if ROC_PATH.exists():
            st.image(str(ROC_PATH), use_container_width=True)
        else:
            st.warning(f"Image not found: {ROC_PATH}")
        st.markdown(
            "<div class='bhi-caption'>Ablation analysis of the surgical edge-strength "
            "model. The combined graph-embedding model distinguished historically strong "
            "surgical problem–solution graph relationships from weaker historical graph "
            "relationships with high accuracy. In this prototype, that signal is used "
            "to prioritize research hypotheses; it does not clinically validate any "
            "individual candidate.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='bhi-footer-note'>The concept maps are exploratory visualizations. "
        "They support hypothesis generation and prioritization, but not clinical decision-making.</div>",
        unsafe_allow_html=True,
    )


def render_about(df: pd.DataFrame, source: str) -> None:
    render_header()

    st.markdown("## About BHI3.1 / BreakingRules")
    st.write(
        "Beyond Human Intuition 3.1 is an experimental research framework for "
        "cross-domain hypothesis generation in visceral and general surgery. "
        "The central idea is to search across large surgical and engineering literatures, "
        "build concept representations, project external engineering mechanisms into "
        "surgical problem–solution spaces, and prioritize candidate research hypotheses "
        "that resemble historically strong surgical innovation patterns."
    )

    st.markdown("### Author")
    st.write(
        "Max F. Stauffer is a medical student from Vienna who aims to become a "
        "general surgeon and software engineer. The project reflects an attempt to combine "
        "clinical surgical thinking, literature mining, machine learning, and software "
        "engineering into a practical hypothesis-generation tool."
    )

    st.markdown("### Current prototype")
    st.write(
        "This webapp does not generate new hypotheses live. It displays a curated prototype "
        "table of pre-generated hypotheses from the BHI3.1 pipeline. This design keeps the "
        "app lightweight, faster to load, and safer for public demonstration."
    )

    st.markdown("### Methodological workflow")
    st.markdown(
        """
1. Build a PubMed-derived surgical corpus and extract surgical problems and solution concepts.
2. Embed surgical problem–solution pairs and construct a historical surgical concept graph.
3. Extract external engineering mechanisms from OpenAlex.
4. Project OpenAlex mechanisms into the surgical solution space.
5. Score candidate problem–mechanism pairs with a surgical edge-strength model.
6. Diversify the high-support candidate pool.
7. Formulate readable research hypotheses using an LLM.
8. Require expert review before any biological, preclinical, clinical, or translational interpretation.
        """
    )

    st.markdown("### GitHub")
    st.write("Repository: https://github.com/maxstaufferferdinand-cmyk/BHI3.0")

    st.markdown("### Data caution")
    st.markdown(
        "<div class='bhi-caution'>"
        "The displayed hypotheses are not medical advice, clinical guidance, or validated "
        "therapeutic concepts. They are intended for research ideation and must be checked "
        "by domain experts before further use."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Loaded app data", expanded=False):
        st.write(f"Rows loaded: **{len(df):,}**")
        st.write(f"Source file: `{source}`")
        st.write(f"Loaded domains: **{df['merged_surgical_domain'].nunique():,}**")
        st.write(f"Loaded problem clusters: **{df['problem_cluster_label'].nunique():,}**")


def main() -> None:
    init_session_state()
    page = render_sidebar()

    try:
        df, source = load_hypotheses()
    except Exception as exc:
        render_header()
        st.error("Could not load hypothesis data.")
        st.exception(exc)
        st.markdown(
            "Expected a CSV file under "
            "`streamlit_prototype_exports/merged_cluster_domains/` with filename pattern "
            "`streamlit_hypotheses_with_merged_domains_*.csv`."
        )
        return

    if page == "Hypothesis generator":
        render_hypothesis_generator(df, source)
    elif page == "Concept graphs":
        render_concept_graphs()
    elif page == "About BHI3.1":
        render_about(df, source)


if __name__ == "__main__":
    main()
