# ML training progress: surgical cluster-edge strength model

This checkpoint documents the first successful machine-learning stage of the BHI3.0 / BreakingRules surgical concept graph pipeline.

## Pipeline steps

1. Built a structured surgical problem-solution graph from extracted PubMed visceral surgery technology abstracts.
2. Aggregated extracted links into exact problem-solution edges.
3. Repaired edge year information from PMID/year metadata.
4. Identified that exact edges were highly fragmented and singleton-dominated.
5. Clustered surgical problem nodes and technical solution nodes using embedding vectors.
6. Aggregated exact edges into semantic cluster-level edges:
   - problem_cluster -> solution_cluster
7. Computed time-adapted cluster-edge strength.
8. Defined a binary ML endpoint:
   - Positive class: top 20% of final cluster-level edge strength
   - Negative class: bottom 50%
   - Middle 30% excluded
9. Trained a cluster-edge strength classifier using graph-neighborhood and centroid-similarity features while excluding direct target-defining variables such as raw cluster weight, exact edge count, and final edge strength.

## Current model result

Cluster-edge strength top-20 model:

- Positive cluster-edges: 11,235
- Negative cluster-edges: 29,077
- ROC-AUC: 0.9712
- Average precision: 0.9591

## Interpretation

This model does not yet claim prospective discovery of new surgical hypotheses. It validates that a machine-learning model can distinguish high-strength surgical problem-solution cluster links from weak links using graph-structural and embedding-derived features.

## Next methodological steps

1. Feature-importance review.
2. Ablation testing:
   - graph-only
   - embedding-only
   - combined graph + embedding
3. Optional Graph Neural Network benchmark.
4. Projection of OpenAlex engineering concepts into surgical solution-cluster space.

## Ablation benchmark

Ablation analysis was performed for the cluster-edge strength model.

Results:

- Combined graph + embedding model: ROC-AUC 0.971, AP 0.959
- Graph topology only: ROC-AUC 0.973, AP 0.961
- Embedding only: ROC-AUC 0.691, AP 0.487
- Degree only: ROC-AUC 0.664, AP 0.465
- Hub-weight only: ROC-AUC 0.843, AP 0.758
- No hub-weight features: ROC-AUC 0.958, AP 0.944

Interpretation: graph-topological features were the strongest predictors, while embedding-only features showed moderate independent signal. The no-hub-weight model retained high performance, indicating that the model is not solely driven by high-frequency hub clusters.
