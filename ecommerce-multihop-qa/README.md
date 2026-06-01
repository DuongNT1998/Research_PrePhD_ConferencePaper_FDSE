# Project Structure

This project is organized as a modular research framework for building an adaptive multi-hop retrieval and reasoning system over Knowledge Graphs for E-commerce Product Question Answering.

The structure is designed to support:

- Large-scale data preprocessing
- Knowledge Graph construction using Neo4j
- Multi-hop retrieval experimentation
- Reinforcement Learning-based graph navigation
- Explainable reasoning path generation
- Reproducible research experiments

---

# Root Directory Structure

```text
ecommerce-multihop-qa/
│
├── data/
├── notebooks/
├── src/
├── neo4j/
├── experiments/
├── outputs/
├── scripts/
├── tests/
├── docs/
│
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

---

# Directory Descriptions

# 1. `data/`

This directory contains all datasets and intermediate processed files.

```text
data/
├── raw/
├── interim/
├── processed/
└── samples/
```

## `data/raw/`

Stores original downloaded datasets without modification.

Examples:

```text
meta_Electronics.json.gz
reviews_Electronics.json.gz
```

Important:
- Never edit files inside this folder
- Treat this directory as immutable source data

---

## `data/interim/`

Contains partially processed or transformed data.

Examples:
- Parsed JSON files
- Cleaned review text
- Intermediate parquet files

Example:

```text
parsed_reviews.parquet
```

---

## `data/processed/`

Contains final structured data ready for Knowledge Graph construction and retrieval experiments.

Examples:

```text
products.csv
reviews.csv
relations.csv
aspects.csv
```

These files are typically used for:
- Neo4j import
- Retrieval indexing
- RL environment initialization

---

## `data/samples/`

Small subsets of data for debugging and rapid prototyping.

Examples:
- 100 products
- 1,000 reviews

Useful for:
- Testing pipelines quickly
- Debugging retrieval logic
- Faster iteration during development

---

# 2. `notebooks/`

Contains Jupyter notebooks for:
- Exploratory Data Analysis (EDA)
- Visualization
- Prototype experiments
- Data inspection

Important:
- Core system logic should NOT be implemented here
- Production code should always be moved into `src/`

---

# 3. `src/`

Main source code directory.

```text
src/
├── config/
├── preprocessing/
├── kg/
├── retrieval/
├── reasoning/
├── rl/
├── llm/
├── evaluation/
├── utils/
└── api/
```

---

# 3.1 `src/config/`

Contains configuration files and experiment settings.

Examples:
- Database configurations
- Hyperparameters
- Retrieval settings
- Training configurations

---

# 3.2 `src/preprocessing/`

Handles all ETL (Extract, Transform, Load) operations.

Example structure:

```text
src/preprocessing/
├── parse_metadata.py
├── parse_reviews.py
├── clean_reviews.py
├── extract_aspects.py
├── build_relations.py
└── generate_qa_pairs.py
```

Responsibilities:
- Parse raw Amazon JSON files
- Clean and normalize text
- Extract aspects and sentiment
- Generate graph relations
- Build structured datasets

---

# 3.3 `src/kg/`

Knowledge Graph construction and graph database interaction layer.

Example structure:

```text
src/kg/
├── schema.py
├── neo4j_client.py
├── import_nodes.py
├── import_edges.py
├── graph_builder.py
└── graph_queries.py
```

Responsibilities:
- Define graph ontology
- Create graph schema
- Import nodes and relationships
- Execute Cypher queries
- Manage Neo4j connections

---

# 3.4 `src/retrieval/`

Core retrieval and graph traversal logic.

Example structure:

```text
src/retrieval/
├── bfs_retriever.py
├── fixed_hop_retriever.py
├── vector_retriever.py
├── graph_retriever.py
├── adaptive_retriever.py
└── stopping_policy.py
```

Responsibilities:
- Baseline retrieval methods
- Multi-hop graph traversal
- Adaptive retrieval strategies
- Dynamic stopping policies
- Retrieval ranking

This module is one of the core research contributions of the project.

---

# 3.5 `src/reasoning/`

Handles reasoning path generation and evidence aggregation.

Example structure:

```text
src/reasoning/
├── path_generator.py
├── path_ranker.py
├── evidence_collector.py
└── explanation_builder.py
```

Responsibilities:
- Generate reasoning paths
- Rank evidence chains
- Build explainable outputs
- Aggregate supporting evidence

---

# 3.6 `src/rl/`

Reinforcement Learning components for adaptive graph navigation.

Example structure:

```text
src/rl/
├── state.py
├── action.py
├── reward.py
├── environment.py
├── policy_network.py
├── trainer.py
└── replay_buffer.py
```

Responsibilities:
- Define RL state/action space
- Reward function design
- Graph navigation environment
- Policy learning
- Adaptive traversal training

---

# 3.7 `src/llm/`

Large Language Model integration layer.

Example structure:

```text
src/llm/
├── prompt_templates.py
├── weak_supervision.py
├── answer_generator.py
└── semantic_evaluator.py
```

Responsibilities:
- Prompt engineering
- Weak supervision generation
- Final answer synthesis
- Semantic evaluation

---

# 3.8 `src/evaluation/`

Contains evaluation metrics and benchmarking code.

Responsibilities:
- Accuracy computation
- Retrieval evaluation
- F1 score calculation
- Explainability metrics
- Ablation studies

---

# 3.9 `src/utils/`

Utility functions shared across the project.

Examples:
- Logging
- File handling
- Serialization
- Embedding helpers
- Common preprocessing functions

---

# 3.10 `src/api/`

Optional API layer for serving the system.

Possible uses:
- FastAPI server
- Retrieval API
- Demo interface
- Interactive QA system

---

# 4. `neo4j/`

Contains Neo4j-related resources.

```text
neo4j/
├── cypher/
├── import/
└── docker/
```

---

## `neo4j/cypher/`

Stores reusable Cypher scripts.

Examples:

```text
create_constraints.cypher
retrieval_queries.cypher
graph_statistics.cypher
```

---

## `neo4j/import/`

Contains CSV files prepared for Neo4j bulk import.

Examples:
- Nodes CSV
- Relationship CSV
- Attribute CSV

---

## `neo4j/docker/`

Docker configuration for Neo4j deployment.

Useful for:
- Reproducible environments
- Easy local setup
- Team collaboration

---

# 5. `experiments/`

Contains experiment configurations and experiment tracking files.

Example structure:

```text
experiments/
├── exp_fixed_hop.yaml
├── exp_rl.yaml
├── exp_stopping.yaml
└── exp_negative_constraints.yaml
```

Responsibilities:
- Store experiment parameters
- Reproduce benchmarks
- Organize ablation studies

---

# 6. `outputs/`

Stores generated outputs from experiments.

```text
outputs/
├── logs/
├── checkpoints/
├── results/
└── figures/
```

---

## `outputs/logs/`

Training and evaluation logs.

---

## `outputs/checkpoints/`

Saved model checkpoints.

Examples:
- RL policy models
- Embedding models

---

## `outputs/results/`

Experimental results and evaluation outputs.

Examples:
- Retrieval metrics
- QA performance reports
- Benchmark summaries

---

## `outputs/figures/`

Generated plots and visualizations for papers or reports.

Examples:
- Retrieval graphs
- Performance curves
- Knowledge Graph visualizations

---

# 7. `scripts/`

Contains executable utility scripts.

Example structure:

```text
scripts/
├── download_data.sh
├── run_preprocessing.sh
├── run_training.sh
└── run_evaluation.sh
```

Responsibilities:
- Automate workflows
- Simplify experiment execution
- Standardize pipeline execution

---

# 8. `tests/`

Contains unit tests and integration tests.

Example structure:

```text
tests/
├── test_parser.py
├── test_neo4j.py
├── test_retrieval.py
└── test_rl_env.py
```

Responsibilities:
- Ensure pipeline correctness
- Validate graph construction
- Test retrieval behavior
- Improve reproducibility

---

# 9. `docs/`

Documentation directory.

Possible contents:
- Architecture diagrams
- Research notes
- Methodology explanations
- Experiment documentation

---

# Configuration Files

# `requirements.txt`

Python dependencies for pip environments.

---

# `environment.yml`

Conda environment configuration.

---

# `.gitignore`

Defines files and directories excluded from Git version control.

Recommended exclusions:

```gitignore
data/
outputs/
__pycache__/
*.pt
*.ckpt
.env
.neo4j/
```

---

# Recommended Development Workflow

1. Download and store raw datasets in `data/raw/`
2. Run preprocessing pipelines
3. Generate structured graph data
4. Import graph into Neo4j
5. Build retrieval baselines
6. Implement adaptive retrieval policies
7. Train RL navigation models
8. Evaluate retrieval and QA performance
9. Generate explainable reasoning paths
10. Conduct ablation and benchmark experiments

---

# Design Philosophy

This structure separates:

- Data engineering
- Knowledge Graph construction
- Retrieval research
- Reinforcement Learning components
- Evaluation and experimentation

The goal is to maintain:
- Modularity
- Reproducibility
- Scalability
- Clean research organization
- Easy experimentation

This organization also makes the project easier to:
- Extend
- Benchmark
- Open-source
- Collaborate on
- Reproduce for academic publications
```