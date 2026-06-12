# BHI3.0 / BreakingRules

BHI3.0 is an experimental research pipeline for cross-domain surgical hypothesis generation. The project explores whether large-scale biomedical and engineering literature can be connected in a structured way to identify plausible, non-obvious surgical research hypotheses. The central idea is not to predict clinical outcomes directly, but to reduce the enormous combinatorial search space between surgical problems and external technical mechanisms into a smaller, machine-ranked and human-reviewable set of candidate hypotheses.

The project was developed around a surgical concept graph and an external engineering mechanism space. On the surgical side, historical surgical literature was processed into problem–solution structures. These structures were then represented as a graph of surgical problem clusters and surgical solution clusters. The graph captures which types of surgical problems have historically been connected to which types of technical or procedural solutions. On the engineering side, OpenAlex-derived engineering mechanisms were extracted, normalized, embedded, and projected into the surgical solution space. This allows external mechanisms to be linked to surgical problem regions through historically learned surgical solution patterns.

The repository contains the code used for the OpenAlex engineering mechanism embedding, surgical–engineering projection, candidate-space estimation, machine-learning-based candidate filtering, diversification, annotation, and LLM-based hypothesis formulation. The PubMed mining and original large-scale surgical literature extraction are not included in this repository. The current repository therefore starts from already prepared surgical graph artifacts and OpenAlex mechanism files. These input artifacts are expected to exist locally and are intentionally not committed to GitHub because they are large generated research data files.

The working directory assumed by most scripts is:

`C:\Users\Max Stauffer\BreakingRules`

Some paths are currently hard-coded for local development. Before using the pipeline on another machine, the base path in the scripts should be adjusted.

The broader methodological logic is as follows. First, a surgical training graph was built from previously extracted surgical problem–solution concepts. Exact surgical problem and solution phrases were clustered into semantic problem and solution clusters. Historical problem–solution cluster edges were then used to train an edge-strength classifier. This model was trained to distinguish historically strong surgical problem–solution edges, especially upper-quintile edges, from weaker historical graph relationships. In prior internal validation, graph-topological features were particularly important, while embedding-only performance was weaker. This supports the interpretation that the pipeline is mainly learning historically successful structural graph patterns, not simply semantic similarity.

The external engineering side was built from OpenAlex mechanism concepts. The file `openalex_engineering_mechanisms_full_fast.csv` contains OpenAlex work identifiers and two extracted mechanism columns, `mechanism_1` and `mechanism_2`. These mechanisms were normalized, deduplicated, filtered for obvious placeholders, embedded with OpenAI embeddings, and stored in a cached embedding directory. The cached embedding script writes progress continuously so that long runs can be resumed after interruption or system failure.

After embedding the OpenAlex mechanisms, the pipeline maps them to surgical solution clusters. This projection is not used as the final hypothesis score. It is only used to connect an external engineering mechanism to the surgical solution-cluster space. Once a mechanism is mapped to one or more surgical solution clusters, the historical surgical graph is used to identify surgical problem clusters connected to those solution clusters. This creates candidate pairs of the form:

OpenAlex engineering mechanism + surgical problem cluster

The first candidate-space estimate showed that the naive Cartesian space between surgical problem clusters and all OpenAlex mechanisms would contain approximately 575.6 million possible pairs. Using the graph-based projection framework reduced this to approximately 101.5 million framework-supported candidate pairs. This was still far too large for direct LLM use, but it was a meaningful reduction of the combinatorial search space.

The next step was strict ML-based filtering. Each candidate inherited the prediction of the underlying mapped surgical problem-cluster to solution-cluster edge. The score was deliberately defined as ML prediction only. OpenAlex-to-surgical similarity was not included in the score. Similarity was used only for the projection step, not for ranking. This was an important methodological decision because the aim was to prioritize candidates by historically learned surgical graph strength rather than by semantic closeness alone.

The ML probability distribution showed a highly top-heavy score structure. Out of 101,490,595 framework-supported candidates, approximately 11.57 million had a predicted probability of at least 0.99 of resembling a historical upper-quintile surgical edge. This indicated that probability thresholding alone was not enough. Therefore, a diversification funnel was applied. The final selection strategy used `p >= 0.99` as a hard ML filter, then kept the top 5 candidates per OpenAlex concept, the top 50 candidates per surgical problem cluster, and the top 500 candidates per mapped surgical solution cluster. This produced a final diversified candidate pool of 29,500 candidates. The final pool contained 20,495 unique OpenAlex concepts, 767 surgical problem clusters, and 662 mapped surgical solution clusters.

The final 29,500 candidates were then annotated with readable problem-cluster labels, solution-cluster labels, representative terms, historical edge metadata, and LLM-ready input fields. From this annotated pool, smaller review files were created, including a Top 1,000 file for manual inspection and a Top 10,000 file for larger downstream LLM runs. A test run of 50 random candidates was created first to inspect the style and usefulness of the generated hypotheses before scaling to the full 29,500-candidate run.

The LLM step formulates one cautious, mechanistic, testable hypothesis per candidate. The prompt explicitly instructs the model not to claim clinical efficacy, not to invent trial results, not to invent animal data, and not to present the output as proven. The intended output is a concise hypothesis title, a 1–2 sentence hypothesis text, a mechanistic rationale, a possible testing route, and a risk or caveat. The LLM output is meant for expert review and prioritization, not as validated scientific evidence.

The most important scripts in the repository are described below in prose rather than as a formal pipeline table.

`embed_openalex_engineering_mechanisms_cached.py` embeds the deduplicated OpenAlex engineering mechanism concepts. It reads the OpenAlex mechanism CSV, extracts unique valid mechanism strings, removes placeholders and missing outputs, and writes an embedding index, a NumPy embedding matrix, a progress file, removed-term logs, and a summary JSON. The key feature of this version is caching: embeddings are written continuously to disk so that progress is not lost after a crash.

`estimate_openalex_hypothesis_candidate_space.py` estimates the size of the OpenAlex-to-surgery candidate universe. It loads the surgical solution-cluster embeddings, the OpenAlex mechanism embeddings, and the historical surgical problem–solution cluster graph. It maps each OpenAlex concept to its top surgical solution clusters and counts how many surgical problem clusters become reachable through those mappings. This script produced the main candidate-space reduction from the naive Cartesian space to the framework-supported candidate set.

`reconstruct_cluster_edge_feature_table_for_ml_scoring.py` reconstructs the historical surgical edge feature table required by the trained edge-strength classifier. This was needed because the model file existed, but the original feature CSV used for scoring was not found in the expected output directory. The script rebuilds features such as centroid cosine, cluster sizes, graph degrees without the current edge, total edge weights without the current edge, exact-edge counts, and density proxy. The reconstructed table is then used for ML scoring.

`score_openalex_ml_prediction_distribution.py` scores the framework-supported candidate space using ML prediction only. It does not include similarity in the score. It calculates the distribution of predicted probabilities and counts how many candidates pass thresholds such as 0.90, 0.95, 0.99, and 0.995. This script showed that roughly 11.57 million candidates had `p >= 0.99`.

`select_diversified_high_ml_openalex_candidates.py` applies the high-probability and diversification funnel. It starts from the OpenAlex-to-surgical-solution mapping and the historical edge ML predictions, keeps only high-ML candidates with `p >= 0.99`, then applies the top-per-OpenAlex, top-per-problem, and top-per-solution diversification strategy. The result is the final 29,500-candidate pool.

`annotate_diversified_candidates_with_cluster_labels.py` adds human-readable cluster annotations to the final candidate pool. It joins problem-cluster terms, solution-cluster terms, and historical edge metadata to the selected candidates. It also creates LLM-ready context fields and exports the annotated full pool, a Top 1,000 manual review file, and a Top 10,000 LLM file.

`llm_formulate_50_random_hypotheses_test.py` performs a small LLM test run on 50 randomly sampled candidates from the Top 1,000 review file. It saves each result immediately, supports resume behavior, and writes both clean CSV output and raw JSONL output. This script is used to inspect whether the hypothesis formulation style is acceptable before running thousands of candidates.

`clean_llm_hypotheses_excel_export.py` converts the 50-candidate test output into a clean Excel workbook. It creates a review sheet with manual review columns, a full-output sheet, and a summary sheet. The goal is to make the generated hypotheses easy to inspect in Excel.

`llm_formulate_full_hypotheses_in_parts.py` is intended for the full LLM run over all 29,500 annotated candidates. It is designed to be run in parts of 5,000 candidates each. It supports long request timeouts, retry/backoff behavior, resume logic, separate part outputs, and a master output file. The current full-run plan is to process final ranks 1–5,000, 5,001–10,000, 10,001–15,000, 15,001–20,000, 20,001–25,000, and 25,001–29,500 as separate runs.

Several plotting scripts are also included. These produce UMAP maps, heatmaps, and combined surgical–OpenAlex concept visualizations. These figures are exploratory and mainly used to understand the structure of the embedded spaces. A key methodological point is that mixed UMAP visualizations must be interpreted carefully. UMAP coordinates are useful for visual neighborhood structure but should not be overinterpreted as direct metric distances. For bridge-building, the projection-based approach is more important than visual overlap in a single UMAP.

The repository intentionally ignores large generated outputs, embeddings, model artifacts, raw LLM outputs, and local secrets. The `.gitignore` excludes files such as `.env`, API-key files, `.npy`, `.npz`, `.pkl`, `.joblib`, cached embedding directories, candidate-output directories, hypothesis-output directories, and other large generated output folders. This is important because the repository is meant to contain reproducible code and lightweight documentation, not large private or generated datasets.

To use the LLM scripts, an OpenAI API key must be set locally in the PowerShell session. The key should never be committed to GitHub. A typical local run uses:

`$env:OPENAI_API_KEY="sk-..."`

`$env:OPENAI_LLM_MODEL="gpt-5.5"`

The full LLM run should be started in parts. For example, the first part is run with:

`python llm_formulate_full_hypotheses_in_parts.py --start 1 --limit 5000 --part-name part01`

The following runs continue with starts at 5001, 10001, 15001, 20001, and 25001. If a run is interrupted, it can be restarted with the same command. Previously completed final ranks are skipped.

This project should be understood as a research prototype. The generated hypotheses are not validated medical recommendations. The ML model identifies candidates that resemble historically strong surgical problem–solution graph patterns. The LLM then converts these structured candidates into readable research hypotheses. Both steps require expert review. Clinical plausibility, safety, feasibility, novelty, and ethical considerations must be assessed separately before any experimental or translational use.

AI disclosure: AI assistance was used during development to adapt code, write documentation, structure the pipeline, and formulate prompts. The scientific interpretation, research direction, and final responsibility for use of the outputs remain with the project author.
