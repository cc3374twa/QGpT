# -*- coding: utf-8 -*-
"""
QGpT: Improving Table Retrieval with Question Generation from Partial Tables
Corpus Embedding Builder

This script builds vector embeddings for table corpora using the QGpT framework.
It processes table data from JSON files and creates vector databases with corpus-specific naming.

Usage:
    python corpus_embedding_builder.py [corpus_file_path]
    python corpus_embedding_builder.py --list  # 列出所有可用語料庫
    python corpus_embedding_builder.py --all   # 建立所有語料庫的 embedding

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel

from utils import (
    load_json_dataset, 
    preprocess_text, 
    extract_corpus_name_from_path,
    generate_db_name,
    generate_collection_name,
    get_corpus_files,
    validate_corpus_structure
)


class CorpusEmbeddingBuilder:
    """語料庫嵌入向量建立器"""
    
    def __init__(self, embedding_dim=1024):
        """
        初始化建立器
        
        Args:
            embedding_dim: 嵌入向量維度 (BGE-M3 輸出 1024 維)
        """
        self.embedding_dim = embedding_dim
        print("🔄 初始化 BGE-M3 模型...")
        self.embedding_fn = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("✅ BGE-M3 模型載入完成")
        
    def build_embeddings(self, corpus_path, force_rebuild=False):
        """
        為指定語料庫建立嵌入向量
        
        Args:
            corpus_path: 語料庫檔案路徑
            force_rebuild: 是否強制重建（即使資料庫已存在）
            
        Returns:
            是否成功建立
        """
        try:
            # 提取語料庫名稱和生成資料庫名稱
            corpus_name = extract_corpus_name_from_path(corpus_path)
            db_name = generate_db_name(corpus_name)
            collection_name = generate_collection_name(corpus_name)
            
            print(f"🔄 處理語料庫: {corpus_name}")
            print(f"   檔案路徑: {corpus_path}")
            print(f"   資料庫名稱: {db_name}")
            print(f"   集合名稱: {collection_name}")
            
            # 檢查資料庫是否已存在
            if Path(db_name).exists() and not force_rebuild:
                print(f"ℹ️  資料庫 '{db_name}' 已存在，跳過建立（使用 --force 強制重建）")
                return True
            
            # 載入語料庫資料
            print("🔄 載入語料庫資料...")
            data = load_json_dataset(corpus_path)
            
            if not validate_corpus_structure(data):
                print(f"❌ 語料庫結構驗證失敗: {corpus_path}")
                return False
            
            print(f"✅ 載入了 {len(data)} 個表格文件")
            
            # 建立 Milvus 向量資料庫客戶端
            print("🔄 初始化向量資料庫...")
            client = MilvusClient(db_name)
            
            # 刪除現有集合（如果存在）並建立新集合
            if client.has_collection(collection_name=collection_name):
                client.drop_collection(collection_name=collection_name)
                print("🗑️  已刪除現有集合")
            
            # 建立新的向量集合
            client.create_collection(
                collection_name=collection_name,
                dimension=self.embedding_dim,
            )
            print("✅ 建立新的向量集合")
            
            # 準備文件資料
            print("🔄 預處理文件...")
            documents = []
            metadata = []
            
            for item in data:
                # 預處理文字
                clean_text = preprocess_text(item['Text'])
                documents.append(clean_text)
                
                # 保存元資料
                metadata.append({
                    'id': item['id'],
                    'filename': item.get('FileName', ''),
                    'sheet_name': item.get('SheetName', '')
                })
            
            # 生成嵌入向量
            print("🔄 生成嵌入向量...")
            vectors = self.embedding_fn.encode(documents)['dense_vecs']  # BGE-M3 使用 encode 方法
            print(f"向量維度: {len(vectors[0])}, 向量數量: {len(vectors)}")
            
            # 準備插入資料
            print("🔄 準備插入資料...")
            insert_data = []
            for i, (doc, vec, meta) in enumerate(zip(documents, vectors, metadata)):
                insert_data.append({
                    "id": i,
                    "vector": vec.astype('float32').tolist(),  # 轉換為 float32 並轉為 list
                    "text": doc,  # 限制文字長度以節省空間
                    "original_id": meta['id'],
                    "filename": meta['filename'],
                    "sheet_name": meta['sheet_name']
                })
            
            # 插入資料到 Milvus
            print("🔄 插入資料到向量資料庫...")
            client.insert(collection_name=collection_name, data=insert_data)
            
            print(f"✅ 成功建立語料庫 '{corpus_name}' 的嵌入向量")
            print(f"   資料庫檔案: {db_name}")
            print(f"   記錄數量: {len(insert_data)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 建立嵌入向量失敗: {e}")
            return False
    
    def build_all_embeddings(self, force_rebuild=False):
        """
        為所有可用語料庫建立嵌入向量
        
        Args:
            force_rebuild: 是否強制重建
            
        Returns:
            建立結果字典 {corpus_name: success}
        """
        corpus_files = get_corpus_files()
        results = {}
        
        if not corpus_files:
            print("⚠️  沒有找到任何語料庫檔案")
            return results
        
        print(f"🔄 找到 {len(corpus_files)} 個語料庫檔案")
        
        for i, corpus_info in enumerate(corpus_files, 1):
            print(f"\n{'='*60}")
            print(f"處理語料庫 {i}/{len(corpus_files)}")
            success = self.build_embeddings(corpus_info['path'], force_rebuild)
            results[corpus_info['name']] = success
        
        # 顯示總結
        print(f"\n{'='*60}")
        print("建立完成總結:")
        successful = sum(1 for success in results.values() if success)
        print(f"成功: {successful}/{len(results)}")
        
        for corpus_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {corpus_name}")
        
        return results


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='QGpT 語料庫嵌入向量建立器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 為指定語料庫建立嵌入向量
  python corpus_embedding_builder.py Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json
  
  # 列出所有可用語料庫
  python corpus_embedding_builder.py --list
  
  # 為所有語料庫建立嵌入向量
  python corpus_embedding_builder.py --all
  
  # 強制重建已存在的資料庫
  python corpus_embedding_builder.py --all --force
        """
    )
    
    parser.add_argument('corpus_path', nargs='?', help='語料庫檔案路徑')
    parser.add_argument('--list', action='store_true', help='列出所有可用語料庫')
    parser.add_argument('--all', action='store_true', help='為所有語料庫建立嵌入向量')
    parser.add_argument('--force', action='store_true', help='強制重建已存在的資料庫')
    parser.add_argument('--dim', type=int, default=1024, help='嵌入向量維度 (預設: 1024)')
    
    args = parser.parse_args()
    
    # 列出所有可用語料庫
    if args.list:
        corpus_files = get_corpus_files()
        if not corpus_files:
            print("沒有找到任何語料庫檔案")
            return
        
        print("可用的語料庫檔案:")
        print("=" * 80)
        for i, corpus_info in enumerate(corpus_files, 1):
            print(f"{i:2d}. 名稱: {corpus_info['name']}")
            print(f"     路徑: {corpus_info['path']}")
            print(f"     資料庫: {corpus_info['db_name']}")
            print(f"     集合: {corpus_info['collection_name']}")
            print()
        return
    
    # 初始化建立器
    builder = CorpusEmbeddingBuilder(embedding_dim=args.dim)
    
    # 為所有語料庫建立嵌入向量
    if args.all:
        builder.build_all_embeddings(force_rebuild=args.force)
        return
    
    # 為指定語料庫建立嵌入向量
    if args.corpus_path:
        if not Path(args.corpus_path).exists():
            print(f"❌ 找不到檔案: {args.corpus_path}")
            sys.exit(1)
        
        success = builder.build_embeddings(args.corpus_path, force_rebuild=args.force)
        sys.exit(0 if success else 1)
    
    # 如果沒有提供參數，顯示幫助
    parser.print_help()


if __name__ == "__main__":
    main()
