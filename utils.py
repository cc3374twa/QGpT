# -*- coding: utf-8 -*-
"""
QGpT: Improving Table Retrieval with Question Generation from Partial Tables
Utility Functions

This module contains common utility functions used across the QGpT project.

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import json
import os
from typing import List, Dict, Tuple
from pathlib import Path


def load_json_dataset(file_path):
    """
    載入 JSON 格式的資料集
    
    Args:
        file_path: JSON 檔案路徑
        
    Returns:
        包含資料的字典列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def preprocess_text(text):
    """
    預處理文字，移除多餘的空格和格式
    
    Args:
        text: 原始文字
        
    Returns:
        清理後的文字
    """
    # 替換 nan 值
    text = text.replace('| nan |', '|  |')
    # 合併多個空格
    text = ' '.join(text.split())
    return text


def extract_corpus_name_from_path(file_path):
    """
    從檔案路徑中提取語料庫名稱，用於資料庫命名
    每個 JSON 檔案對應一個獨立的資料庫
    
    Args:
        file_path: 語料庫檔案路徑
        
    Returns:
        語料庫名稱（用於資料庫命名）
    """
    path_obj = Path(file_path)
    
    # 如果是在 Corpora 目錄下
    if 'Corpora' in path_obj.parts:
        corpora_index = path_obj.parts.index('Corpora')
        name_parts = []
        
        # 收集 Corpora 後的所有目錄名稱和檔案名稱（不含副檔名）
        for i in range(corpora_index + 1, len(path_obj.parts)):
            part = path_obj.parts[i]
            # 如果是最後一個部分（檔案名），去掉副檔名
            if i == len(path_obj.parts) - 1:
                part = Path(part).stem
            name_parts.append(part)
        
        return '_'.join(name_parts)
    
    # 如果不在 Corpora 目錄下，使用檔案名（不含副檔名）
    return path_obj.stem


def generate_db_name(corpus_name):
    """
    根據語料庫名稱生成資料庫檔案名稱
    
    Args:
        corpus_name: 語料庫名稱
        
    Returns:
        資料庫檔案名稱
    """
    # 清理名稱，替換特殊字符
    clean_name = corpus_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    
    # 縮短名稱以符合 Milvus 36 字符限制
    # 移除常見的重複詞彙並使用簡寫
    clean_name = clean_name.replace('Table', 'T')
    clean_name = clean_name.replace('mimo_table_length_variation', 'MTLV')
    clean_name = clean_name.replace('Single_Table_Retrieval', 'STR')
    clean_name = clean_name.replace('Multi_Table_Retrieval', 'MTR')
    clean_name = clean_name.replace('table_representation', 'TR')
    
    db_name = f"qgpt_{clean_name}.db"
    
    # 如果仍然太長，進一步縮短
    if len(db_name) > 35:  # 留一個字符的緩衝
        # 移除底線並只保留前30個字符
        short_name = clean_name.replace('_', '')[:25]
        db_name = f"qgpt_{short_name}.db"
    
    return db_name


def generate_collection_name(corpus_name):
    """
    根據語料庫名稱生成集合名稱
    
    Args:
        corpus_name: 語料庫名稱
        
    Returns:
        向量集合名稱
    """
    # 清理名稱，替換特殊字符
    clean_name = corpus_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    
    # 縮短名稱以符合命名限制
    clean_name = clean_name.replace('Table', 'T')
    clean_name = clean_name.replace('mimo_table_length_variation', 'MTLV')
    clean_name = clean_name.replace('Single_Table_Retrieval', 'STR')
    clean_name = clean_name.replace('Multi_Table_Retrieval', 'MTR')
    clean_name = clean_name.replace('table_representation', 'TR')
    
    collection_name = f"emb_{clean_name}"
    
    # 如果仍然太長，進一步縮短
    if len(collection_name) > 35:
        # 移除底線並只保留前30個字符
        short_name = clean_name.replace('_', '')[:30]
        collection_name = f"emb_{short_name}"
    
    return collection_name


def get_corpus_files(corpora_dir="Corpora"):
    """
    取得 Corpora 目錄下的所有語料庫檔案
    
    Args:
        corpora_dir: Corpora 目錄路徑
        
    Returns:
        語料庫檔案資訊列表，包含 path, name, db_name, collection_name
    """
    corpus_files = []
    corpora_path = Path(corpora_dir)
    
    if not corpora_path.exists():
        print(f"警告：找不到 Corpora 目錄：{corpora_dir}")
        return []
    
    # 遞迴搜尋所有 JSON 檔案
    for json_file in corpora_path.rglob("*.json"):
        corpus_name = extract_corpus_name_from_path(str(json_file))
        db_name = generate_db_name(corpus_name)
        collection_name = generate_collection_name(corpus_name)
        
        corpus_files.append({
            'path': str(json_file),
            'name': corpus_name,
            'db_name': db_name,
            'collection_name': collection_name
        })
    
    return corpus_files


def format_search_results(results, query, format_type="detailed"):
    """
    格式化搜索結果
    
    Args:
        results: 搜索結果列表
        query: 搜索查詢
        format_type: 格式類型 ("detailed", "simple", "json")
        
    Returns:
        格式化後的結果字符串
    """
    if not results:
        return "沒有找到相關結果"
    
    if format_type == "simple":
        # 簡單格式輸出
        output = f"查詢: {query}\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['filename']} | {result['sheet_name']} | 相似度: {result['score']:.4f}\n"
        return output.strip()
    
    elif format_type == "json":
        # JSON 格式輸出
        output_dict = {
            "query": query,
            "results": [
                {
                    "rank": i+1,
                    "score": result['score'],
                    "filename": result['filename'],
                    "sheet_name": result['sheet_name'],
                    "original_id": result.get('original_id', '')
                }
                for i, result in enumerate(results)
            ]
        }
        return json.dumps(output_dict, ensure_ascii=False, indent=2)
    
    else:
        # 詳細格式輸出
        output = f"搜索查詢: '{query}'\n"
        output += f"找到 {len(results)} 個相關結果:\n"
        output += "=" * 80 + "\n"
        
        for i, result in enumerate(results, 1):
            output += f"\n結果 {i} (相似度: {result['score']:.4f})\n"
            output += f"檔案: {result['filename']}\n"
            output += f"工作表: {result['sheet_name']}\n"
            output += f"原始ID: {result.get('original_id', 'N/A')}\n"
            output += f"內容預覽:\n"
            
            # 顯示內容預覽
            content_lines = result['text'].split('\n')[:3]
            for line in content_lines:
                if line.strip():
                    preview = line[:100] + ('...' if len(line) > 100 else '')
                    output += f"  {preview}\n"
            
            if len(content_lines) < len(result['text'].split('\n')):
                output += "  ...\n"
            
            output += "-" * 60 + "\n"
        
        return output.strip()


def validate_corpus_structure(data):
    """
    驗證語料庫資料結構是否正確
    
    Args:
        data: 語料庫數據
        
    Returns:
        是否符合預期結構
    """
    if not data:
        return False
    
    required_fields = ['Text', 'id']
    
    for item in data:
        if not isinstance(item, dict):
            return False
        
        for field in required_fields:
            if field not in item:
                print(f"警告：缺少必要欄位 '{field}'")
                return False
    
    return True
