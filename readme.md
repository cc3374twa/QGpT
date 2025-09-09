# QGpT: Improving Table Retrieval with Question Generation from Partial Tables

This repository contains the source code, corpora, and model prompts for the paper:  
📄 [**Improving Table Retrieval with Question Generation from Partial Tables**](https://openreview.net/forum?id=Q8HOV0UMwA) (TRL Workshop @ ACL 2025)


> We propose **QGpT**, a simple yet effective framework that improves open-domain table retrieval by generating synthetic questions from partial tables.

---

<img width="961" height="496" alt="image" src="https://github.com/user-attachments/assets/5cbf5a1f-0af7-48bb-976d-b3c086162117" />


## 🆕 Latest Updates

### 🔥 New (25/09/09)
- **Complete System Refactor**: Modular architecture with improved code organization
- **New Components**: Added `query_evaluator.py`, `utils.py`, and `demo_restructured.py`
- **Enhanced Features**: Batch processing, intelligent database naming, comprehensive evaluation
- **Better User Experience**: Interactive interfaces, detailed error messages, and auto-detection

### 📊 Dataset Release (25/09/08)

Question-to-Gold-Table datasets for recall@k evaluation are available in the `test_dataset/` folder of this repository.

This dataset contains question-table pairs from the original datasets, and is structured into multiple test files:

Available test datasets:
- **E2E-WTQ**: `test_dataset/E2E-WTQ_test.json`
- **FeTAQA**: `test_dataset/FetaQA_test.json`
- **MMQA 2tables**: `test_dataset/MMQA-2tables_test.json`
- **MMQA 3tables**: `test_dataset/MMQA-3tables_test.json`
- **MimoTable Chinese**: `test_dataset/MimoTable-Chinese_test.json`
- **MimoTable English**: `test_dataset/MimoTable-English_test.json`
- **OTTQA**: `test_dataset/OTT-QA_test.json`


➡️ **Alternative Source:** [cc3374twa/QGPT](https://huggingface.co/datasets/cc3374twa/QGPT) (Hugging Face Dataset)

## 🗂️ Repository Contents

- 📁 `Corpora/`  
  Experimental datasets corresponding to the tables reported in the paper. Each subfolder contains the exact data used for specific experiments:
  - `Table1_mimo_table_length_variation/` → **Table 1** experiments (table length variation analysis)
  - `Table3_mimo_en_table_representation/` → **Table 3** experiments (table representation methods)
  - `Table5_Single_Table_Retrieval/` → **Table 5** experiments (single table retrieval evaluation)
  - `Table6_Multi_Table_Retrieval/` → **Table 6** experiments (multi-table retrieval evaluation)
  - `Table7_OTTQA/` → **Table 7** experiments (OTTQA dataset evaluation)

- 📁 `test_dataset/`  
  Question-to-Gold-Table datasets for recall@k evaluation across different benchmarks:
  - E2E-WTQ, FeTAQA, MMQA (2tables/3tables), MimoTable (Chinese/English), OTTQA

- 📁 `prompt/`  
  Prompt templates for question generation and query decomposition (MMQA).
  - `llama3-8b-Instruct_header_extract_and_QG.txt`
  - `llama3-8b-Instruct_QG_only.txt`
  - `MMQA_query_decomposition.txt`

---

## 📦 Embedding Model & Vector Database

The table corpora under `Corpora/` are processed using our proposed QGpT method and converted into vector embeddings for efficient retrieval.

**Embedding Models Supported:**
- Dense embeddings via **Milvus** vector database → [https://milvus.io](https://milvus.io)  
- Sparse embeddings via **RAGatouille** (ColBERT) → [https://github.com/AnswerDotAI/RAGatouille](https://github.com/AnswerDotAI/RAGatouille)

**Current Implementation:**
- **Default Model**: Milvus with 768-dimensional dense vectors
- **Embedding Generation**: Automated via `corpus_embedding_builder.py`
- **Storage Format**: Local Milvus database files (`.db`)

---

## 🧪 Reproducibility

- All table corpora are constructed based on the same datasets used in the paper.
- Each folder maps to the exact experimental tables (e.g., Table 1, Table 5).

⚠️ If any released data differs from what’s reported in the paper due to human error, please contact us at [cc3374twa@gmail.com](mailto:cc3374twa@gmail.com).

---

## � Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- `pymilvus[model]>=2.3.0` - Milvus vector database client
- `numpy>=1.21.0` - Numerical computing support

### Usage

#### 1. Run Demo
```bash
python demo_restructured.py
```
This demonstrates the complete workflow including corpus listing, embedding generation, and sample queries.

#### 2. Build Embeddings

**List all available corpora:**
```bash
python corpus_embedding_builder.py --list
```

**Build embeddings for a specific corpus:**
```bash
python corpus_embedding_builder.py Corpora/Table1_mimo_table_length_variation/mimo_en/1k_token.json
```

**Build embeddings for all corpora:**
```bash
python corpus_embedding_builder.py --all
```

#### 3. Search Tables

**Basic search (auto-detect database):**
```bash
python qgpt_search.py "financial statements"
```

**Advanced search with options:**
```bash
# Specify output format and limit
python qgpt_search.py "construction project" -n 10 -f json

# Use specific database
python qgpt_search.py "student data" --db qgpt_Table1_mimo_en.db
```

**List available databases:**
```bash
python qgpt_search.py --list-dbs
```

#### 4. Query Evaluation

**Single query evaluation:**
```bash
python query_evaluator.py "financial data" --db qgpt_Table5_Single_Table_Retrieval_QGpT.db
```

**Batch evaluation with test files:**
```bash
python query_evaluator.py --test-file Test_Query_and_GroundTruth_Table/E2E-WTQ_test.json --db qgpt_Table5_Single_Table_Retrieval_QGpT.db
```

**Batch evaluation for all test sets:**
```bash
python query_evaluator.py --batch-eval
```

## 🏗️ System Architecture

### Database Naming Convention

The system automatically generates database and collection names based on corpus paths:

```
Corpus Path → Database Name
Corpora/Table1_mimo_table_length_variation/mimo_en/1k_token.json 
→ qgpt_Table1_mimo_table_length_variation_mimo_en.db

Collection Name
Table1_mimo_table_length_variation_mimo_en 
→ embeddings_Table1_mimo_table_length_variation_mimo_en
```

### Core Components

1. **`corpus_embedding_builder.py`** - Corpus embedding generator
2. **`qgpt_search.py`** - Command-line search interface  
3. **`query_evaluator.py`** - Query evaluation and testing
4. **`utils.py`** - Common utility functions
5. **`demo_restructured.py`** - Complete system demonstration

---
## � Supported Datasets

The system supports the following experimental corpora from the paper:

- **Table 1**: `Table1_mimo_table_length_variation/` - Table length variation experiments
- **Table 3**: `Table3_mimo_en_table_representation/` - Table representation methods  
- **Table 5**: `Table5_Single_Table_Retrieval/` - Single table retrieval experiments
- **Table 6**: `Table6_Multi_Table_Retrieval/` - Multi-table retrieval experiments
- **Table 7**: `Table7_OTTQA/` - OTTQA dataset experiments

Each corpus can be processed independently with automatic database naming and collection management.

## 🔧 Technical Features

- **Vector Dimension**: 768D (configurable)
- **Search Speed**: Millisecond-level response
- **Similarity Range**: 0.0 - 1.0 (higher = more similar)
- **Language Support**: Chinese, English, and mixed queries
- **Database Format**: Milvus vector database (.db files)
- **Batch Processing**: Support for processing all corpora at once
- **Evaluation Support**: Built-in evaluation against ground truth

## 📞 Contact

For questions or issues, please contact: [cc3374twa@gmail.com](mailto:cc3374twa@gmail.com)

---

## �📄 Citation

If you find this repository or its data useful, citing our paper would be appreciated:

```bibtex
@inproceedings{
liang2025improving,
title={Improving Table Retrieval with Question Generation from Partial Tables},
author={Hsing-Ping Liang and Che-Wei Chang and Yao-Chung Fan},
booktitle={The 4th Table Representation Learning Workshop at ACL 2025},
year={2025},
url={https://openreview.net/forum?id=Q8HOV0UMwA}
}
