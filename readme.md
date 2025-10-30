# QGpT: Improving Table Retrieval with Question Generation from Partial Tables

This repository contains the source code, corpora, and model prompts for the paper:  
📄 [**Improving Table Retrieval with Question Generation from Partial Tables**](https://openreview.net/forum?id=Q8HOV0UMwA) (TRL Workshop @ ACL 2025)


> We propose **QGpT**, a simple yet effective framework that improves open-domain table retrieval by generating synthetic questions from partial tables.

---

## 🆕 Updates

### 2025/10/30

We have released **Milvus-Lite implementation** for local vector database operations:

- ✅ Complete Python scripts for embedding and indexing table corpora
- ✅ Search and evaluation pipeline with Recall@k metrics
- ✅ Jupyter notebooks with interactive examples
- ✅ Conda environment configuration (`Milvus.yml`)

This allows you to reproduce our experiments locally without requiring a Milvus server deployment.

### 2025/08/08

We have released a original Question to gold-table datasets that we used to test recall@k:

➡️ **Hugging Face Dataset:** [cc3374twa/QGPT](https://huggingface.co/datasets/cc3374twa/QGPT)

This dataset contains question-table pairs from the original datasets, and is structured into multiple subsets (e.g., `E2E-WTQ`).

To load the dataset using `datasets`:

```python
from datasets import load_dataset

# Load the E2E-WTQ subset
dataset = load_dataset("cc3374twa/QGPT", name="E2E-WTQ", split="test")
```

## 🗂️ Repository Contents

- 📁 `Corpora/`  
  Table corpora used in each experimental table in the paper. Each subfolder corresponds to one experiment section:
  - `Table1_mimo_table_length_variation/`
  - `Table3_mimo_en_table_representation/`
  - `Table5_Single_Table_Retrieval/`
  - `Table6_Multi_Table_Retrieval/`
  - `Table7_OTTQA/`

- 📁 `prompt/`  
  Prompt templates for question generation and query decomposition (MMQA).
  - `llama3-8b-Instruct_header_extract_and_QG.txt`
  - `llama3-8b-Instruct_QG_only.txt`
  - `MMQA_query_decomposition.txt`

---

## 📦 Dataset Construction

The table corpora under `Corpora/` are preprocessed and embedded based on our proposed method.  
They are indexed using either:

- **Milvus** → [https://milvus.io](https://milvus.io)  
- **RAGatouille** → [https://github.com/AnswerDotAI/RAGatouille](https://github.com/AnswerDotAI/RAGatouille)

### 🗄️ Milvus Implementation

We provide Python scripts for working with **Milvus-Lite** (single-file vector database):

- 📁 `embedding_db/pymilvus/`  
  - `create_pymilvus_db.py` - Create Milvus collections from table corpora
  - `pymilvus_embedding.py` - Embed and insert data using BAAI/bge-m3 model

- 📁 `evaluation/`  
  - `search_pymilvus.py` - Search queries in Milvus collections
  - `evaluation.py` - Evaluate retrieval performance with Recall@k metrics

- 📓 Jupyter Notebooks:
  - `Milvus.ipynb` - Complete workflow for Milvus database creation and embedding
  - `E2EWTQ_Milvus_search.ipynb` - E2E-WTQ dataset search example
  - `export_milvus.py` - Export metadata from Milvus collections

- ⚙️ Environment:
  - `Milvus.yml` - Conda environment specification with all dependencies

---

## 🧪 Reproducibility

- All table corpora are constructed based on the same datasets used in the paper.
- Each folder maps to the exact experimental tables (e.g., Table 1, Table 5).

⚠️ If any released data differs from what’s reported in the paper due to human error, please contact us at [cc3374twa@gmail.com](mailto:cc3374twa@gmail.com).

---

## 🚀 Quick Start

### Setup Milvus Environment

```bash
# Create conda environment from yml file
conda env create -f Milvus.yml
conda activate Milvus
```

### Create Vector Database

```bash
cd embedding_db/pymilvus
python create_pymilvus_db.py  # Create collections
python pymilvus_embedding.py   # Embed and insert data
```

### Search and Evaluate

```bash
cd evaluation
python search_pymilvus.py      # Perform vector search
python evaluation.py           # Calculate Recall@k
```

## 🚧 TODO

- [ ] Add setup instructions for RAGatouille
- [ ] Complete evaluation for all datasets

---
## 📄 Citation

If you find this repository or its data useful, citing our paper would be appreciated:

ACL :
```
Hsing-Ping Liang, Che-Wei Chang, and Yao-Chung Fan. 2025. Improving Table Retrieval with Question Generation from Partial Tables. In Proceedings of the 4th Table Representation Learning Workshop, pages 217–228, Vienna, Austria. Association for Computational Linguistics.
```
Bibtex :
```bibtex
@inproceedings{liang-etal-2025-improving-table,
    title = "Improving Table Retrieval with Question Generation from Partial Tables",
    author = "Liang, Hsing-Ping  and
      Chang, Che-Wei  and
      Fan, Yao-Chung",
    booktitle = "Proceedings of the 4th Table Representation Learning Workshop",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.trl-1.19/",
    doi = "10.18653/v1/2025.trl-1.19",
    pages = "217--228",
}