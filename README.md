# BHI3.0 / BreakingRules

BHI3.0 is a research prototype for graph-based, cross-domain hypothesis generation in surgery. The project connects historical surgical problem–solution patterns with external engineering mechanisms derived from OpenAlex in order to generate mechanistic, testable surgical research hypotheses.

The core idea is to reduce the very large combinatorial search space between surgical problems and technical concepts into a smaller, machine-ranked, diversified, and human-reviewable candidate set. The system does not predict clinical outcomes and does not claim that generated hypotheses are clinically valid. It is a literature-based discovery and prioritization pipeline intended for expert review.

## Project status

This repository contains the current OpenAlex-to-surgery hypothesis generation pipeline. It includes code for embedding engineering mechanisms, estimating candidate-space size, scoring candidates with a trained surgical graph model, selecting a diversified high-confidence candidate pool, annotating candidates with readable cluster labels, and formulating hypotheses with an LLM.

The repository does not contain the original PubMed mining code or the full raw literature-mining datasets. It assumes that surgical graph artifacts and OpenAlex mechanism files have already been generated locally.

Current processed candidate counts from the latest run:

* Naive surgical problem cluster × OpenAlex concept space: approximately 575.6 million pairs
* Framework-supported candidate space after OpenAlex-to-surgical-solution projection: 101,490,595 pairs
* Candidates with ML probability ≥ 0.99: 11,574,071 pairs
* Final diversified candidate pool: 29,500 pairs
* Final pool coverage: 20,495 OpenAlex concepts, 767 surgical problem clusters, and 662 mapped surgical solution clusters

## Repository scope

This repository is intended to store reproducible pipeline code and lightweight documentation. Large intermediate files, embeddings, model binaries, raw LLM outputs, and generated data tables are intentionally excluded from GitHub.

Included:

* OpenAlex mechanism embedding code
* Candidate-space estimation code
* Surgical edge-feature reconstruction code
* ML probability distribution analysis
* High-ML candidate diversification
* Candidate annotation scripts
* LLM test and full-run scripts
* Plotting scripts for concept-space visualization
* README and project documentation

Not included:

* Original PubMed mining pipeline
* Full raw PubMed abstract datasets
* Full OpenAlex raw datasets
* Large `.npy` embedding matrices
* Trained `.joblib` model files
* Large generated CSV outputs
* API keys or local secrets

## Method overview

The pipeline starts from a previously generated surgical concept graph. Surgical literature was represented as problem–solution structures. Similar surgical problems and solutions were clustered into problem clusters and solution clusters. Historical problem–solution cluster edges were then used to train a classifier that distinguishes strong historical surgical graph relationships from weaker ones.

OpenAlex engineering mechanisms are processed separately. Extracted mechanism strings are normalized, deduplicated, embedded, and projected into the surgical solution-cluster space. This projection is used only to connect external engineering mechanisms to potentially relevant historical surgical solution regions.

A candidate hypothesis is then defined as:

`OpenAlex engineering mechanism + surgical problem cluster`

The candidate exists if the OpenAlex mechanism maps to a surgical solution cluster that has historical graph links to surgical problem clusters. The ML model then scores the corresponding surgical problem-cluster to solution-cluster relationship. The score used in the current selection funnel is strictly the ML probability that this mapped surgical problem–solution configuration resembles a historical upper-quintile surgical edge.

OpenAlex-to-surgical similarity is not used as a ranking score. Similarity is only used for the projection step.

## Selection funnel

The current selection funnel is:

1. Generate framework-supported candidates through OpenAlex mechanism to surgical solution-cluster projection.
2. Assign each candidate the ML probability of the corresponding surgical problem-cluster to solution-cluster relationship.
3. Apply a strict threshold of `ml_probability_top20_like >= 0.99`.
4. Keep the top 5 candidates per OpenAlex concept.
5. Keep the top 50 candidates per surgical problem cluster.
6. Keep the top 500 candidates per mapped surgical solution cluster.
7. Export the final diversified candidate pool.

This produces the file:

`openalex_high_ml_diversified_candidates_outputs/final_diversified_high_ml_openalex_candidates.csv`

The annotated version is generated later as:

`openalex_hypothesis_annotation_outputs/final_diversified_high_ml_openalex_candidates_annotated.csv`

## Main input files

The scripts assume the following local input files exist.

Surgical graph files:

`C:\Users\Max Stauffer\BreakingRules\surgical_time_normalized_cluster_edges\cluster_edges_time_normalized.csv`

`C:\Users\Max Stauffer\BreakingRules\surgical_time_normalized_cluster_edges\problem_node_clusters.csv`

`C:\Users\Max Stauffer\BreakingRules\surgical_time_normalized_cluster_edges\solution_node_clusters.csv`

Surgical embedding files:

`C:\Users\Max Stauffer\BreakingRules\surgical_training_graph_outputs\surgical_problem_embedding_index.csv`

`C:\Users\Max Stauffer\BreakingRules\surgical_training_graph_outputs\surgical_solution_embedding_index.csv`

`C:\Users\Max Stauffer\BreakingRules\surgical_training_graph_outputs\surgical_problem_embeddings.npy`

`C:\Users\Max Stauffer\BreakingRules\surgical_training_graph_outputs\surgical_solution_embeddings.npy`

`C:\Users\Max Stauffer\BreakingRules\surgical_training_graph_outputs\surgical_pair_embeddings.npy`

OpenAlex mechanism file:

`C:\Users\Max Stauffer\BreakingRules\openalex_engineering_mechanisms_full_fast.csv`

The OpenAlex mechanism file is expected to contain at least:

* `openalex_id`
* `mechanism_1`
* `mechanism_2`

The repository currently uses hard-coded local paths. For use on another machine, update `BASE_DIR` in the scripts.

## Main scripts

### `embed_openalex_engineering_mechanisms_cached.py`

Embeds OpenAlex engineering mechanisms. The script extracts unique mechanism strings from `mechanism_1` and `mechanism_2`, removes obvious placeholder terms, and writes a cached embedding matrix.

Main outputs:

`openalex_engineering_mechanism_embeddings_cached/openalex_engineering_mechanism_index.csv`

`openalex_engineering_mechanism_embeddings_cached/openalex_engineering_mechanism_embeddings.npy`

`openalex_engineering_mechanism_embeddings_cached/openalex_engineering_mechanism_progress.csv`

`openalex_engineering_mechanism_embeddings_cached/openalex_engineering_mechanism_summary.json`

The cached version is preferred because it writes progress after each batch and can recover after crashes.

### `estimate_openalex_hypothesis_candidate_space.py`

Estimates the candidate space created by mapping OpenAlex mechanisms to surgical solution clusters and then expanding through historical surgical problem-cluster links.

Main outputs:

`hypothesis_candidate_space_outputs/candidate_space_summary.json`

`hypothesis_candidate_space_outputs/candidate_space_funnel.csv`

`hypothesis_candidate_space_outputs/openalex_concept_candidate_counts.csv`

`hypothesis_candidate_space_outputs/problem_cluster_candidate_counts.csv`

`hypothesis_candidate_space_outputs/openalex_to_surgical_solution_cluster_topk.csv`

The top-k mapping file is later used by the scoring and selection scripts.

### `reconstruct_cluster_edge_feature_table_for_ml_scoring.py`

Reconstructs the feature table needed to score historical surgical problem-cluster to solution-cluster edges with the trained ML model.

The reconstructed features include:

* `centroid_cosine`
* `problem_cluster_size_log`
* `solution_cluster_size_log`
* `problem_degree_without_edge_log`
* `solution_degree_without_edge_log`
* `problem_total_weight_without_edge_log`
* `solution_total_weight_without_edge_log`
* `problem_total_exact_edges_without_edge_log`
* `solution_total_exact_edges_without_edge_log`
* `cluster_edge_density_proxy`

Main output:

`cluster_edge_strength_model_outputs/reconstructed_cluster_edge_feature_table_for_ml_scoring.csv`

### `score_openalex_ml_prediction_distribution.py`

Scores the OpenAlex-mediated candidate space using the trained surgical edge-strength model. The candidate score is the ML prediction only. No OpenAlex-to-surgical similarity term is used in the score.

Main outputs:

`openalex_ml_prediction_distribution_outputs/historical_cluster_edge_ml_predictions.csv`

`openalex_ml_prediction_distribution_outputs/openalex_candidate_ml_prediction_distribution.csv`

`openalex_ml_prediction_distribution_outputs/openalex_candidate_ml_prediction_threshold_counts.csv`

`openalex_ml_prediction_distribution_outputs/top_ml_predicted_openalex_problem_candidates_preview.csv`

`openalex_ml_prediction_distribution_outputs/openalex_candidate_ml_prediction_summary.json`

### `select_diversified_high_ml_openalex_candidates.py`

Applies the final candidate selection funnel. It keeps candidates with `ml_probability_top20_like >= 0.99` and then applies top-k diversification across OpenAlex concepts, surgical problem clusters, and mapped surgical solution clusters.

Main outputs:

`openalex_high_ml_diversified_candidates_outputs/stage1_p99_top5_per_openalex_concept.csv`

`openalex_high_ml_diversified_candidates_outputs/stage2_p99_top50_per_problem_cluster.csv`

`openalex_high_ml_diversified_candidates_outputs/stage3_p99_top500_per_solution_cluster.csv`

`openalex_high_ml_diversified_candidates_outputs/final_diversified_high_ml_openalex_candidates.csv`

`openalex_high_ml_diversified_candidates_outputs/diversified_high_ml_candidate_selection_summary.json`

### `annotate_diversified_candidates_with_cluster_labels.py`

Adds readable cluster labels, representative cluster terms, and historical edge metadata to the selected candidate pool.

Main outputs:

`openalex_hypothesis_annotation_outputs/final_diversified_high_ml_openalex_candidates_annotated.csv`

`openalex_hypothesis_annotation_outputs/top1000_for_manual_review.csv`

`openalex_hypothesis_annotation_outputs/top10000_for_llm.csv`

`openalex_hypothesis_annotation_outputs/candidate_annotation_summary.json`

### `llm_formulate_50_random_hypotheses_test.py`

Runs a small LLM test on 50 randomly sampled candidates from the Top 1,000 manual review file. This is used to inspect hypothesis style before running the full candidate pool.

Main outputs:

`openalex_llm_hypothesis_test_outputs/llm_test_50_random_input_candidates.csv`

`openalex_llm_hypothesis_test_outputs/llm_test_50_random_hypotheses.csv`

`openalex_llm_hypothesis_test_outputs/llm_test_50_random_raw_outputs.jsonl`

`openalex_llm_hypothesis_test_outputs/llm_test_50_random_summary.json`

### `clean_llm_hypotheses_excel_export.py`

Converts the 50-candidate LLM test output into a clean Excel workbook for manual review.

Main output:

`openalex_llm_hypothesis_test_outputs/llm_test_50_random_hypotheses_CLEAN.xlsx`

### `llm_formulate_full_hypotheses_in_parts.py`

Runs the LLM hypothesis formulation over the full annotated candidate pool. The script is designed to run in parts of 5,000 candidates and supports retries, long request timeouts, backoff pauses, part-level outputs, master outputs, and resume logic.

The intended run structure is:

`--start 1 --limit 5000 --part-name part01`

`--start 5001 --limit 5000 --part-name part02`

`--start 10001 --limit 5000 --part-name part03`

`--start 15001 --limit 5000 --part-name part04`

`--start 20001 --limit 5000 --part-name part05`

`--start 25001 --limit 5000 --part-name part06`

Main outputs:

`openalex_llm_hypothesis_full_outputs/full_llm_hypotheses_part01.csv`

`openalex_llm_hypothesis_full_outputs/full_llm_hypotheses_part02.csv`

`openalex_llm_hypothesis_full_outputs/full_llm_hypotheses_MASTER.csv`

`openalex_llm_hypothesis_full_outputs/full_llm_hypotheses_MASTER_raw.jsonl`

## Plotting scripts

The repository also contains scripts for exploratory visualization:

`plot_openalex_umap_heatmap.py`

`plot_openalex_umap_labeled.py`

`plot_openalex_umap_labeled_adjusted.py`

`plot_combined_surgical_openalex_concept_map.py`

`plot_surgical_pair_map_with_projected_openalex.py`

These scripts create UMAP maps, labeled concept maps, heatmaps, and projected OpenAlex-to-surgery visualizations. These figures are useful for understanding the structure of the embedding space, but they should not be interpreted as direct proof of hypothesis quality.

## Installation

The project uses Python 3.14 in the current local setup.

Typical packages include:

`pandas`

`numpy`

`scikit-learn`

`joblib`

`openai`

`openpyxl`

`matplotlib`

`umap-learn`

`adjustText`

Install dependencies as needed:

`python -m pip install pandas numpy scikit-learn joblib openai openpyxl matplotlib umap-learn adjustText`

A formal `requirements.txt` should be added once the environment is finalized.

## API key setup

The LLM and embedding scripts require an OpenAI API key. The key must be set locally and must not be committed to GitHub.

PowerShell example:

`$env:OPENAI_API_KEY="sk-..."`

`$env:OPENAI_LLM_MODEL="gpt-5.5"`

If a placeholder such as `DEIN_API_KEY_HIER_EINFÜGEN` is used by mistake, the scripts may fail because the placeholder contains non-ASCII characters. Use the real API key only.

## Running the pipeline

The typical local order is:

1. Embed OpenAlex engineering mechanisms.
2. Estimate the candidate space.
3. Reconstruct surgical edge features if needed.
4. Score the candidate probability distribution.
5. Select diversified high-ML candidates.
6. Annotate selected candidates.
7. Run a 50-candidate LLM test.
8. Review the Excel output.
9. Run the full LLM pipeline in 5,000-candidate parts.

Example commands:

`python embed_openalex_engineering_mechanisms_cached.py`

`python estimate_openalex_hypothesis_candidate_space.py`

`python reconstruct_cluster_edge_feature_table_for_ml_scoring.py`

`python score_openalex_ml_prediction_distribution.py`

`python select_diversified_high_ml_openalex_candidates.py`

`python annotate_diversified_candidates_with_cluster_labels.py`

`python llm_formulate_50_random_hypotheses_test.py`

`python clean_llm_hypotheses_excel_export.py`

`python llm_formulate_full_hypotheses_in_parts.py --start 1 --limit 5000 --part-name part01`

## Output policy

Large generated files are excluded from version control. This includes embeddings, model files, large candidate outputs, raw LLM outputs, and local cache directories.

The `.gitignore` should exclude:

`*.npy`

`*.npz`

`*.pkl`

`*.joblib`

`.env`

`openalex_engineering_mechanism_embeddings_cached/`

`hypothesis_candidate_space_outputs/`

`openalex_ml_prediction_distribution_outputs/`

`openalex_high_ml_diversified_candidates_outputs/`

`openalex_hypothesis_annotation_outputs/`

`openalex_llm_hypothesis_test_outputs/`

`openalex_llm_hypothesis_full_outputs/`

This keeps the repository focused on code and documentation rather than generated data.

## Scientific interpretation

The generated hypotheses are not clinical recommendations. The pipeline identifies OpenAlex-mediated problem–mechanism candidates that map onto historically strong surgical graph regions. The ML score indicates similarity to historical high-strength surgical problem–solution graph patterns, not clinical truth.

All generated hypotheses require expert review for surgical plausibility, novelty, feasibility, safety, ethics, and translational relevance before any further use.

## AI disclosure

AI assistance was used to adapt code, structure scripts, write documentation, and support prompt development. The outputs of the pipeline and the generated hypotheses require expert scientific review. Responsibility for interpretation and use remains with the project author.

## Citation

Citation will be added once available
