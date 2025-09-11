# QGpT: Improving Table Retrieval with Question Generation from Partial Tables

This repository contains the source code, corpora, and model prompts for the paper:  
📄 [**Improving Table Retrieval with Question Generation from Partial Tables**](https://openreview.net/forum?id=Q8HOV0UMwA) (TRL Workshop @ ACL 2025)

> We propose **QGpT**, a simple yet effective framework that improves open-domain table retrieval by generating synthetic questions from partial tables.

---

<img width="961" height="496" alt="image" src="https://github.com/user-attachments/assets/5cbf5a1f-0af7-48bb-976d-b3c086162117" />

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
- **Primary Model**: **BGE-M3** (BAAI/bge-m3) with 1024-dimensional dense vectors
- **Model Features**: 
  - Multi-language support (Chinese/English)
  - FP16 precision for efficiency
  - Superior semantic understanding
- **Embedding Generation**: Automated via `corpus_embedding_builder.py`
- **Storage Format**: Local Milvus database files (`.db`) with intelligent naming

---

## 🧪 Reproducibility

- All table corpora are constructed based on the same datasets used in the paper.
- Each folder maps to the exact experimental tables (e.g., Table 1, Table 5).

⚠️ If any released data differs from what's reported in the paper due to human error, please contact us at [cc3374twa@gmail.com](mailto:cc3374twa@gmail.com).

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Main Dependencies:**
- `pymilvus[model]>=2.3.0` - Milvus vector database client
- `FlagEmbedding` - BGE-M3 embedding model
- `numpy>=1.21.0` - Numerical computing support

### Usage

#### 1. Build Embeddings

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

#### 2. Query Evaluation

**Single query evaluation:**
```bash
python query_evaluator.py "Q: who is the best performance user" --db qgpt_T5SingleTRetrievalQGpTE2E.db
```

**Batch evaluation with test files:**
```bash
python query_evaluator.py --test-file test_dataset/E2E-WTQ_test.json --db qgpt_T5SingleTRetrievalQGpTE2E.db

# Chinese dataset evaluation
python query_evaluator.py --test-file test_dataset/MimoTable-Chinese_test.json --db qgpt_T1_MTLV_mimo_ch_1k_token.db
```

**Batch evaluation for all test sets:**
```bash
python query_evaluator.py --batch-eval
```

## 🏗️ System Architecture

### Database Naming Convention

The system automatically generates database and collection names based on corpus paths, with **each JSON file corresponding to an independent database**:

```
Corpus Path → Database Name (includes complete file path)
Corpora/Table1_mimo_table_length_variation/mimo_en/1k_token.json 
→ qgpt_T1_MTLV_mimo_en_1k_token.db

Collection Name
Table1_mimo_table_length_variation_mimo_en_1k_token 
→ emb_T1_MTLV_mimo_en_1k_token
```

#### Name Simplification Rules:
- `Table` → `T` (simplify table numbers)
- `mimo_table_length_variation` → `MTLV`
- `Single_Table_Retrieval` → `STR`
- `Multi_Table_Retrieval` → `MTR`
- `table_representation` → `TR`

#### Example Mappings:
```
File Name                                         Database Name
1k_token.json                            → qgpt_T1_MTLV_mimo_en_1k_token.db
2k_token.json                            → qgpt_T1_MTLV_mimo_en_2k_token.db
5k_token.json                            → qgpt_T1_MTLV_mimo_en_5k_token.db
Full-Table(8k).json                      → qgpt_T1_MTLV_mimo_en_Full-T(8k).db
E2EWTQ_QGpT.json                         → qgpt_T5SingleTRetrievalQGpTE2E.db
pT.json                                  → qgpt_T3_mimo_en_TR_pT.db

Currently Available Databases (30+ total):
- Table 1 (Length Variation): qgpt_T1_MTLV_mimo_ch/en_*.db (8 databases)
- Table 3 (Representation): qgpt_T3_mimo_en_TR_*.db (5 databases)
- Table 5 (Single Retrieval): qgpt_T5SingleTRetrieval*.db (8 databases)
- Table 6 (Multi Retrieval): qgpt_T6MultiTRetrieval*.db (2 databases)
- Table 7 (OTTQA): qgpt_T7*.db (3 databases)
```

#### Database Isolation Benefits:
- **Precise Control**: Each corpus file is managed and queried independently
- **Avoid Confusion**: Data from different experimental conditions is completely isolated
- **Flexible Operations**: Individual datasets can be rebuilt, queried, or deleted separately
- **Version Management**: Different versions of the same concept (e.g., different token lengths) are independent

### Core Components

1. **`corpus_embedding_builder.py`** - Corpus embedding generator
2. **`qgpt_search.py`** - Command-line search interface  
3. **`query_evaluator.py`** - Query evaluation and testing
4. **`utils.py`** - Common utility functions

---

## 📊 Supported Datasets

The system supports the following experimental corpora from the paper:

- **Table 1**: `Table1_mimo_table_length_variation/` - Table length variation experiments
- **Table 3**: `Table3_mimo_en_table_representation/` - Table representation methods  
- **Table 5**: `Table5_Single_Table_Retrieval/` - Single table retrieval experiments
- **Table 6**: `Table6_Multi_Table_Retrieval/` - Multi-table retrieval experiments
- **Table 7**: `Table7_OTTQA/` - OTTQA dataset experiments

Each corpus can be processed independently with automatic database naming and collection management.

## 🔧 Technical Features

- **Embedding Model**: BGE-M3 (BAAI/bge-m3) with FP16 precision
- **Vector Dimension**: 1024D (upgraded from 768D)
- **Search Speed**: Millisecond-level response
- **Similarity Range**: 0.0 - 1.0 (higher = more similar)
- **Language Support**: Enhanced Chinese, English, and mixed queries
- **Performance**: 77.22% Hit Rate@10 on MimoTable-English test set (641 queries)
- **Database Coverage**: 30+ pre-built databases covering all experimental conditions
- **Database Format**: Milvus vector database (.db files) with intelligent path-based naming
- **Batch Processing**: Support for processing all corpora at once
- **Evaluation Support**: Built-in evaluation against ground truth with comprehensive metrics
- **Production Ready**: All experimental data pre-processed and immediately available

## 📞 Contact

For questions or issues, please contact: [cc3374twa@gmail.com](mailto:cc3374twa@gmail.com)

---

## 📄 Citation

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
```
