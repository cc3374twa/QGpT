# -*- coding: utf-8 -*-
"""
QGpT: Improving Table Retrieval with Question Generation from Partial Tables
Command-Line Search Interface

This script provides a command-line interface for searching table corpora
using the QGpT framework with vector embeddings.

Usage:
    python qgpt_search.py "your query" [options]

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import json
import numpy as np
from pymilvus import MilvusClient, model
from typing import List, Dict
import sys
import argparse
from pathlib import Path

from utils import (
    format_search_results,
    get_corpus_files,
    generate_collection_name
)

class QGpTSearchEngine:
    def __init__(self, db_path: str = None, collection_name: str = None):
        """
        初始化 QGpT 表格搜索引擎
        
        Args:
            db_path: Milvus 資料庫檔案路徑
            collection_name: 向量集合名稱
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.embedding_fn = None
        
        # 如果沒有指定資料庫，嘗試自動找到
        if not self.db_path:
            self.db_path = self.find_default_database()
        
        # 如果沒有指定集合名稱，從資料庫路徑推導
        if not self.collection_name and self.db_path:
            corpus_name = Path(self.db_path).stem.replace('qgpt_', '')
            self.collection_name = generate_collection_name(corpus_name)
        
        self.initialize()
    
    def find_default_database(self) -> str:
        """尋找預設的資料庫檔案"""
        # 搜尋當前目錄下的 qgpt_*.db 檔案
        db_files = list(Path('.').glob('qgpt_*.db'))
        
        if not db_files:
            print("❌ 找不到任何 QGpT 資料庫檔案")
            print("請先運行 corpus_embedding_builder.py 來建立資料庫")
            sys.exit(1)
        
        if len(db_files) == 1:
            return str(db_files[0])
        
        # 如果有多個資料庫，讓使用者選擇
        print("找到多個資料庫檔案:")
        for i, db_file in enumerate(db_files, 1):
            print(f"  {i}. {db_file}")
        
        while True:
            try:
                choice = input(f"請選擇資料庫 (1-{len(db_files)}): ")
                index = int(choice) - 1
                if 0 <= index < len(db_files):
                    return str(db_files[index])
                else:
                    print("無效的選擇，請重新輸入")
            except (ValueError, KeyboardInterrupt):
                print("\n已取消操作")
                sys.exit(1)
    
    def initialize(self):
        """初始化 Milvus 客戶端和嵌入函數"""
        try:
            if not Path(self.db_path).exists():
                print(f"❌ 找不到資料庫檔案: {self.db_path}")
                print("請先運行 corpus_embedding_builder.py 來建立資料庫")
                sys.exit(1)
            
            self.client = MilvusClient(self.db_path)
            self.embedding_fn = model.DefaultEmbeddingFunction()
            
            # 檢查集合是否存在
            if not self.client.has_collection(collection_name=self.collection_name):
                print(f"❌ 錯誤：找不到集合 '{self.collection_name}' 在資料庫 '{self.db_path}'")
                print("請先運行 corpus_embedding_builder.py 來建立資料庫")
                sys.exit(1)
            
            print(f"✅ 連接到資料庫: {self.db_path}")
            print(f"✅ 使用集合: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ 初始化失敗: {e}")
            sys.exit(1)
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        搜索相關的表格資料
        
        Args:
            query: 搜索查詢字符串
            limit: 返回結果的數量
            
        Returns:
            搜索結果列表
        """
        try:
            # 將查詢轉換為向量
            query_vector = self.embedding_fn.encode_queries([query])
            
            # 執行搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=query_vector,
                limit=limit,
                output_fields=["text", "filename", "sheet_name", "original_id"]
            )
            
            # 格式化結果
            formatted_results = []
            for result in search_results[0]:
                formatted_results.append({
                    'score': 1 - result['distance'],  # 轉換為相似度分數
                    'filename': result['entity']['filename'],
                    'sheet_name': result['entity']['sheet_name'],
                    'original_id': result['entity']['original_id'],
                    'text': result['entity']['text']
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"搜索錯誤: {e}")
            return []
    
    def display_results(self, results: List[Dict], query: str, format_type: str = "detailed"):
        """顯示搜索結果"""
        output = format_search_results(results, query, format_type)
        print(output)

def main():
    """
    QGpT 命令行搜索主程式
    
    提供命令行介面來搜索 QGpT 表格資料庫
    """
    parser = argparse.ArgumentParser(
        description='QGpT 表格檢索系統 - 使用向量嵌入進行語義搜索',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python qgpt_search.py "財務報表"
  python qgpt_search.py "construction project" -n 10 -f json
  python qgpt_search.py "學生成績" -f simple
        """
    )
    parser.add_argument('query', nargs='?', help='搜索查詢字符串（中文或英文）')
    parser.add_argument('-n', '--limit', type=int, default=5, help='返回結果數量 (預設: 5)')
    parser.add_argument('-f', '--format', choices=['detailed', 'simple', 'json'], 
                       default='detailed', help='輸出格式 (預設: detailed)')
    parser.add_argument('--db', help='QGpT 資料庫路徑（自動偵測如未指定）')
    parser.add_argument('--collection', help='向量集合名稱（自動從資料庫名稱推導）')
    parser.add_argument('--list-dbs', action='store_true', help='列出所有可用的資料庫檔案')
    
    args = parser.parse_args()
    
    # 列出所有可用的資料庫檔案
    if args.list_dbs:
        db_files = list(Path('.').glob('qgpt_*.db'))
        if not db_files:
            print("沒有找到任何 QGpT 資料庫檔案")
            return
        
        print("可用的 QGpT 資料庫檔案:")
        for i, db_file in enumerate(db_files, 1):
            corpus_name = db_file.stem.replace('qgpt_', '')
            collection_name = generate_collection_name(corpus_name)
            print(f"  {i}. {db_file}")
            print(f"     語料庫: {corpus_name}")
            print(f"     集合: {collection_name}")
            print()
        return
    
    # 檢查是否提供了查詢參數
    if not args.query:
        print("❌ 請提供搜索查詢字符串")
        parser.print_help()
        return
    
    # 初始化 QGpT 搜索引擎
    search_engine = QGpTSearchEngine(db_path=args.db, collection_name=args.collection)
    
    # 執行搜索
    results = search_engine.search(args.query, limit=args.limit)
    
    # 顯示結果
    search_engine.display_results(results, args.query, format_type=args.format)

if __name__ == "__main__":
    main()
