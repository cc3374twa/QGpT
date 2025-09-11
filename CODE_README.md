# QGpT: Implementation Code Documentation

This directory contains the complete implementation code for the QGpT framework with a modular architecture.

## 🏗️ Architecture Overview

The system has been refactored into distinct modules with clear responsibilities:

### Core Modules

1. **`utils.py`** - Common utility functions
   - Data loading and preprocessing
   - Corpus name extraction and database naming
   - Result formatting utilities
   - Structure validation

2. **`corpus_embedding_builder.py`** - Corpus embedding generator
   - Build embedding databases for specified corpora
   - Support batch processing for all corpora
   - Automatic database naming based on corpus paths
   - Force rebuild capabilities

3. **`query_evaluator.py`** - Query evaluation engine
   - Single query testing
   - Batch evaluation with ground truth comparison
   - Support for test file processing
   - Performance metrics calculation

4. **`qgpt_search.py`** - Command-line search interface (updated)
   - Intelligent database detection
   - Multi-database support
   - Flexible search interface with multiple output formats

## 📁 Files Description

### Core Implementation
- `corpus_embedding_builder.py` - Main embedding generation script for table corpora
- `qgpt_search.py` - Command-line search interface for querying embedded tables
- `query_evaluator.py` - Query evaluation and testing framework
- `utils.py` - Shared utility functions and helpers
- `example_retrieval.py` - Legacy example script

### Database Files
- `qgpt_*.db` - Milvus vector database files with corpus-specific naming

### Test Datasets
- `test_dataset/` - Question-to-Gold-Table datasets for recall@k evaluation
  - E2E-WTQ, FeTAQA, MMQA (2tables/3tables), MimoTable (Chinese/English), OTTQA

### Configuration & Documentation
- `requirements.txt` - Python dependencies
- `使用指南.md` - Chinese user guide
- `readme.md` - English documentation
- `UPDATED_README.md` - Architecture update documentation

## 🚀 Usage Examples

### 1. Generate Embeddings

**List available corpora:**
```bash
python corpus_embedding_builder.py --list
```

**Build specific corpus:**
```bash
python corpus_embedding_builder.py Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json
```

**Build all corpora:**
```bash
python corpus_embedding_builder.py --all
```

**Force rebuild:**
```bash
python corpus_embedding_builder.py --all --force
```

### 2. Search Operations

**Auto-detect database:**
```bash
python qgpt_search.py "financial statements"
```

**Specify database and format:**
```bash
python qgpt_search.py "財務報表" --db qgpt_Table1_mimo_ch.db --format json
```

**List available databases:**
```bash
python qgpt_search.py --list-dbs
```

### 3. Query Evaluation

**Single query evaluation:**
```bash
python query_evaluator.py "test query" --db qgpt_Table1_mimo_en.db
```

**Batch evaluation:**
```bash
python query_evaluator.py --batch-eval --save
```

**Test file evaluation:**
```bash
python query_evaluator.py --test-file test_dataset/E2E-WTQ_test.json --db qgpt_specific.db
```

## 🔧 Database Naming Convention

⚠️ **Updated (2025/09/09)**: Each JSON file now corresponds to an independent database.

The system uses intelligent naming based on complete corpus paths:

```
Corpus Path → Database Name (includes filename)
Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json
→ qgpt_T1_MTLV_mimo_ch_1k_token.db

Corpora/Table1_mimo_table_length_variation/mimo_ch/2k_token.json
→ qgpt_T1_MTLV_mimo_ch_2k_token.db

Collection Names
Table1_mimo_table_length_variation_mimo_ch_1k_token
→ emb_T1_MTLV_mimo_ch_1k_token
```

### Naming Simplification Rules
- `Table` → `T` (simplify table identifiers)
- `mimo_table_length_variation` → `MTLV`
- `Single_Table_Retrieval` → `STR`
- `Multi_Table_Retrieval` → `MTR`
- `table_representation` → `TR`

This ensures each corpus file has its own isolated database for precise control and management.

## 📋 Dependencies

Core dependencies from `requirements.txt`:
- `pymilvus[model]>=2.3.0` - Milvus vector database client
- `numpy>=1.21.0` - Numerical computing

Optional dependencies for development:
- `transformers>=4.20.0` - For enhanced text processing
- `torch>=1.12.0` - PyTorch backend
- `jupyter>=1.0.0` - Notebook environment

## 🎯 Key Improvements

1. **Modular Design**: Clear separation of concerns across modules
2. **Intelligent Naming**: Auto-generated database names prevent conflicts  
3. **Batch Processing**: Support for processing all corpora at once
4. **Comprehensive Evaluation**: Built-in evaluation framework with metrics
5. **User-Friendly Interface**: Interactive selection and detailed error messages
6. **Flexible Configuration**: Configurable vector dimensions and search parameters

## 📝 Implementation Notes

This implementation corresponds to the table retrieval experiments described in the paper, focusing on:
- Milvus-based vector indexing for efficient similarity search
- Corpus-specific database management for organized data handling
- Comprehensive evaluation framework for reproducible results
- Support for both Chinese and English text processing

The modular architecture allows for easy extension and maintenance while preserving the experimental setup described in the research paper.
