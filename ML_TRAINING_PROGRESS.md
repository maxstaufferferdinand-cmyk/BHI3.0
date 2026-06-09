# ML training progress: surgical cluster-edge strength model

This checkpoint documents the first successful machine-learning stage of the BHI3.0 / BreakingRules surgical concept graph pipeline.

## Pipeline steps

1. Built a structured surgical problem-solution training graph from extracted PubMed visceral surgery technology abstracts.
2. Aggregated extracted links into exact problem-solution edges.
3. Repaired edge year information from source PMID/year metadata.
4. Identified that exact edges were highly fragmented and singleton-dominated.
5. Clustered surgical problem nodes and technical solution nodes using their embedding vectors.
6. Aggregated exact edges into semantic cluster-level edges:
   - problem_cluster -> solution_cluster
7. Computed time-adapted cluster-edge strength.
8. Defined a binary ML endpoint:
   - Positive class: top 20% of cluster-level final edge strength
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

The next methodological steps are:

1. Feature-importance review.
2. Ablation testing:
   - graph-only
   - embedding-only
   - combined graph + embedding
3. Optional Graph Neural Network benchmark.
4. Projection of OpenAlex engineering concepts into surgical solution-cluster space.
