# QGpT 程式架構說明

本專案已重整程式架構，將 embedding 建立與查詢評估分離，並提供以語料庫名稱為基礎的資料庫命名系統。

## 程式檔案架構

### 核心模組

1. **`utils.py`** - 公用工具函數
   - 資料載入與預處理
   - 語料庫名稱提取與資料庫命名
   - 結果格式化工具

2. **`corpus_embedding_builder.py`** - 語料庫嵌入向量建立器
   - 為指定語料庫建立 embedding 資料庫
   - 支援批次建立所有語料庫
   - 自動以語料庫名稱命名資料庫檔案

3. **`query_evaluator.py`** - 查詢評估器
   - 單一查詢測試
   - 批次測試查詢評估
   - 支援 ground truth 比較

4. **`qgpt_search.py`** - 命令行搜索介面（已更新）
   - 智慧資料庫偵測
   - 多資料庫支援
   - 彈性搜索介面

## 資料庫命名規則

資料庫檔案會根據語料庫路徑自動命名：

```
語料庫路徑 → 資料庫名稱
Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json → qgpt_Table1_mimo_table_length_variation_mimo_ch.db
Corpora/Table5_Single_Table_Retrieval/QGpT/E2EWTQ_QGpT.json → qgpt_Table5_Single_Table_Retrieval_QGpT.db
```

向量集合名稱規則：
```
語料庫名稱 → 集合名稱
Table1_mimo_table_length_variation_mimo_ch → embeddings_Table1_mimo_table_length_variation_mimo_ch
```

## 使用說明

### 1. 建立語料庫嵌入向量

#### 為單一語料庫建立 embedding：
```bash
python corpus_embedding_builder.py Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json
```

#### 列出所有可用語料庫：
```bash
python corpus_embedding_builder.py --list
```

#### 為所有語料庫建立 embedding：
```bash
python corpus_embedding_builder.py --all
```

#### 強制重建已存在的資料庫：
```bash
python corpus_embedding_builder.py --all --force
```

### 2. 搜索查詢

#### 自動偵測資料庫進行搜索：
```bash
python qgpt_search.py "財務報表"
```

#### 指定資料庫搜索：
```bash
python qgpt_search.py "construction project" --db qgpt_Table1_mimo_en.db
```

#### 列出所有可用資料庫：
```bash
python qgpt_search.py --list-dbs
```

#### 不同輸出格式：
```bash
# 簡單格式
python qgpt_search.py "學生成績" --format simple

# JSON 格式
python qgpt_search.py "學生成績" --format json
```

### 3. 查詢評估

#### 單一查詢評估：
```bash
python query_evaluator.py "財務報表" --db qgpt_Table1_mimo_ch.db
```

#### 使用測試檔案評估：
```bash
python query_evaluator.py --test-file Test_Query_and_GroundTruth_Table/MiMoTable-English_test.json --db qgpt_Table1_mimo_en.db
```

#### 批次評估所有測試集：
```bash
python query_evaluator.py --batch-eval
```

#### 批次評估並儲存結果：
```bash
python query_evaluator.py --batch-eval --limit 10 --save
```

## 主要改進

1. **清晰的職責分離**：
   - `corpus_embedding_builder.py`：專責建立 embedding
   - `query_evaluator.py`：專責查詢評估
   - `qgpt_search.py`：專責搜索介面

2. **智慧命名系統**：
   - 資料庫檔案自動以語料庫名稱命名
   - 向量集合名稱對應語料庫結構
   - 避免命名衝突

3. **彈性使用介面**：
   - 支援單一或批次操作
   - 自動偵測可用資料庫
   - 多種輸出格式選擇

4. **完整評估功能**：
   - 支援 ground truth 比較
   - 計算命中率和精確率
   - 批次評估所有測試集

5. **錯誤處理與使用者友善**：
   - 詳細的錯誤訊息
   - 進度顯示
   - 互動式選擇介面

## 檔案對應關係

| 原始檔案 | 新檔案 | 功能 |
|---------|--------|------|
| `qgpt_embedding.py` | `corpus_embedding_builder.py` | 建立語料庫 embedding |
| `qgpt_search.py` | `qgpt_search.py` (更新) | 搜索介面 |
| - | `query_evaluator.py` | 查詢評估 |
| - | `utils.py` | 公用工具 |

使用此重整後的架構，您可以更有組織地管理不同語料庫的 embedding，並進行系統性的查詢評估。
