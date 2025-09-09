# QGpT: Improving Table Retrieval with Question Generation from Partial Tables

This directory contains the implementation code for the QGpT framework.

## Files Description

### Core Implementation
- `corpus_embedding_builder.py` - Main embedding generation script for table corpora
- `qgpt_search.py` - Command-line search interface for querying embedded tables
- `example_retrieval.py` - Example script demonstrating retrieval functionality

### Database
- `qgpt_*.db` - Milvus vector database files containing embedded table corpora

## Usage

### 1. Generate Embeddings
```bash
python corpus_embedding_builder.py
```

### 2. Search Tables (Command Line)
```bash
# Basic search
python qgpt_search.py "財務報表"

# With options
python qgpt_search.py "financial statements" -n 10 -f json
```



## Dependencies
- pymilvus[model]
- numpy
- json

## Note
This implementation corresponds to the table retrieval experiments described in the paper, particularly focusing on the Milvus-based indexing approach mentioned in the Dataset Construction section.
