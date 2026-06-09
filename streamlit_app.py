import streamlit as st

st.set_page_config(
    page_title="Beyond Human Intuition V3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("BHI3.0")
st.sidebar.caption("Beyond Human Intuition V3.0")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Surgical concept graph",
        "Hypothesis generator",
        "OpenAlex engineering space",
        "About",
    ],
)

st.sidebar.divider()

surgical_domain = st.sidebar.selectbox(
    "Surgical domain",
    [
        "All domains",
        "Upper GI surgery",
        "Hepatobiliary surgery",
        "Pancreatic surgery",
        "Colorectal surgery",
        "Hernia / abdominal wall",
        "Bariatric surgery",
        "Therapeutic endoscopy",
    ],
)

technology_domain = st.sidebar.selectbox(
    "Technology domain",
    [
        "All technologies",
        "Endoscopy",
        "Robotics",
        "Imaging / navigation",
        "Stents / drainage",
        "Mesh / biomaterials",
        "Energy / ablation",
        "Stapling / anastomosis",
        "Sensors / pressure / flow",
    ],
)

novelty_level = st.sidebar.slider(
    "Novelty level",
    min_value=1,
    max_value=5,
    value=3,
)

st.sidebar.info("Prototype interface only. No machine-learning model is connected yet.")

# Header
st.title("Beyond Human Intuition V3.0")
st.subheader("Surgical Technology Hypothesis Generator")

st.markdown(
    """
    **BHI3.0** is a cross-domain literature-mining framework designed to identify
    novel bridge hypotheses between visceral surgery and engineering concepts.
    """
)

st.divider()

# Pages
if page == "Overview":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Surgical problem nodes", "53,763")
    with col2:
        st.metric("Solution nodes", "81,683")
    with col3:
        st.metric("Problem–solution pairs", "120,798")

    st.subheader("Project workflow")

    st.markdown(
        """
        1. **PubMed surgical corpus**  
           Extract surgical problems and technical/procedural solutions from visceral surgery literature.

        2. **Surgical neural concept graph**  
           Build a graph of historical problem–solution relations using embeddings and publication-level edges.

        3. **Engineering concept projection**  
           Project OpenAlex/arXiv engineering mechanisms into the surgical solution space.

        4. **Hypothesis generation**  
           Rank novel surgery–engineering bridge hypotheses based on similarity, novelty, and plausibility.
        """
    )

    st.success("Current stage: surgical training graph and concept-map visualization completed.")

elif page == "Surgical concept graph":
    st.subheader("Surgical concept graph")

    st.markdown(
        """
        This page will later display the UMAP concept map and graph statistics.

        Planned features:
        - Interactive UMAP of surgical problem–solution pairs
        - Filter by surgical domain
        - Filter by technology type
        - Show high-frequency historical problem–solution relations
        - Export selected graph regions
        """
    )

elif page == "Hypothesis generator":
    st.subheader("Hypothesis generator")

    st.markdown(
        """
        This page will later generate candidate hypotheses by linking surgical problems
        to external engineering mechanisms.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_problem = st.selectbox(
            "Select surgical problem",
            [
                "Anastomotic leakage",
                "Bile duct obstruction",
                "Postoperative pancreatic fistula",
                "Incisional hernia",
                "Colorectal polyp",
                "Gastroesophageal reflux",
            ],
        )

    with col2:
        selected_mode = st.selectbox(
            "Generation mode",
            [
                "Conservative",
                "Balanced",
                "High novelty",
                "Ultra-far bridge",
            ],
        )

    st.button("Generate hypothesis", disabled=True)
    st.warning("Model not connected yet. This button is intentionally disabled.")

elif page == "OpenAlex engineering space":
    st.subheader("OpenAlex engineering space")

    st.markdown(
        """
        This page will later display engineering mechanisms extracted from OpenAlex/arXiv.

        Planned mechanism domains:
        - Microfluidics
        - Soft robotics
        - Continuum robotics
        - Sensors
        - Pressure and flow control
        - Bioadhesive hydrogels
        - Sealing and leakage systems
        - Materials engineering
        """
    )

    st.info("Engineering projection will be added after the surgical graph is finalized.")

elif page == "About":
    st.subheader("About Beyond Human Intuition V3.0")

    st.markdown(
        """
        **Beyond Human Intuition V3.0** aims to support surgical hypothesis generation
        by combining large-scale biomedical literature mining, engineering concept extraction,
        semantic embeddings, and graph-based representation learning.

        The current prototype is focused on visceral surgery and technical/procedural
        innovation. It is intended for research ideation, not clinical decision-making.
        """
    )

    st.caption("Prototype version: 0.1")
