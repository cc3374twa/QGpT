# -*- coding: utf-8 -*-
"""
QGpT: Improving Table Retrieval with Question Generation from Partial Tables
Query Evaluator

This script evaluates query performance using test queries against built vector databases.
It supports both single query testing and batch evaluation with ground truth comparison.

Usage:
    python query_evaluator.py "query text" --db database.db
    python query_evaluator.py --test-file test_queries.json --db database.db
    python query_evaluator.py --batch-eval  # 批次評估所有測試集

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pymilvus import MilvusClient, model

from utils import (
    load_json_dataset,
    format_search_results,
    get_corpus_files,
    extract_corpus_name_from_path,
    generate_db_name,
    generate_collection_name
)


class QGpTQueryEvaluator:
    """QGpT 查詢評估器"""
    
    def __init__(self, db_path: str, collection_name: str):
        """
        初始化查詢評估器
        
        Args:
            db_path: 向量資料庫路徑
            collection_name: 向量集合名稱
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.embedding_fn = None
        self.initialize()
    
    def initialize(self):
        """初始化 Milvus 客戶端和嵌入函數"""
        try:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(f"找不到資料庫檔案: {self.db_path}")
            
            self.client = MilvusClient(self.db_path)
            self.embedding_fn = model.DefaultEmbeddingFunction()
            
            # 檢查集合是否存在
            if not self.client.has_collection(collection_name=self.collection_name):
                raise ValueError(f"找不到集合 '{self.collection_name}' 在資料庫 '{self.db_path}'")
            
            print(f"✅ 成功連接到資料庫: {self.db_path}")
            print(f"✅ 使用集合: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ 初始化失敗: {e}")
            sys.exit(1)
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        執行搜索查詢
        
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
                    'distance': result['distance'],
                    'filename': result['entity']['filename'],
                    'sheet_name': result['entity']['sheet_name'],
                    'original_id': result['entity']['original_id'],
                    'text': result['entity']['text']
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 搜索錯誤: {e}")
            return []
    
    def evaluate_single_query(self, query: str, ground_truth: Optional[List[str]] = None, 
                            limit: int = 5) -> Dict:
        """
        評估單一查詢
        
        Args:
            query: 查詢字符串
            ground_truth: 正確答案列表（檔案名或ID）
            limit: 返回結果數量
            
        Returns:
            評估結果
        """
        results = self.search(query, limit)
        
        evaluation = {
            'query': query,
            'results_count': len(results),
            'results': results
        }
        
        # 如果有正確答案，計算準確率指標
        if ground_truth:
            retrieved_ids = [r['original_id'] for r in results]
            retrieved_files = [r['filename'] for r in results]
            
            # 計算命中率（Hit Rate）
            hits_by_id = sum(1 for gt in ground_truth if gt in retrieved_ids)
            hits_by_file = sum(1 for gt in ground_truth if gt in retrieved_files)
            
            evaluation.update({
                'ground_truth': ground_truth,
                'hits_by_id': hits_by_id,
                'hits_by_file': hits_by_file,
                'hit_rate_by_id': hits_by_id / len(ground_truth) if ground_truth else 0,
                'hit_rate_by_file': hits_by_file / len(ground_truth) if ground_truth else 0,
                'precision_at_k': hits_by_id / len(results) if results else 0
            })
        
        return evaluation


class BatchEvaluator:
    """批次評估器"""
    
    def __init__(self):
        self.test_files_dir = "Test_Query_and_GroundTruth_Table"
    
    def get_test_files(self) -> List[str]:
        """取得所有測試檔案"""
        test_dir = Path(self.test_files_dir)
        if not test_dir.exists():
            return []
        
        return [str(f) for f in test_dir.glob("*.json")]
    
    def match_corpus_to_test(self, test_file: str) -> Optional[str]:
        """
        將測試檔案匹配到對應的語料庫
        
        Args:
            test_file: 測試檔案路徑
            
        Returns:
            對應的語料庫資料庫路徑，如果找不到則返回 None
        """
        test_name = Path(test_file).stem
        
        # 定義測試檔案到語料庫的映射規則
        mapping_rules = {
            'MiMoTable-English': 'Table1_mimo_table_length_variation_mimo_en',
            'MiMoTable-Chinese': 'Table1_mimo_table_length_variation_mimo_ch',
            'E2E-WTQ': 'Table5_Single_Table_Retrieval_QGpT',
            'FetaQA': 'Table5_Single_Table_Retrieval_QGpT',
            'OTT-QA': 'Table7_OTTQA',
            'MMQA-2tables': 'Table6_Multi_Table_Retrieval_2_tables',
            'MMQA-3tables': 'Table6_Multi_Table_Retrieval_3_tables'
        }
        
        # 尋找匹配的語料庫
        for test_key, corpus_pattern in mapping_rules.items():
            if test_key in test_name:
                # 尋找對應的資料庫檔案
                corpus_files = get_corpus_files()
                for corpus_info in corpus_files:
                    if corpus_pattern in corpus_info['name']:
                        db_path = corpus_info['db_name']
                        if Path(db_path).exists():
                            return db_path
        
        return None
    
    def evaluate_test_file(self, test_file: str, db_path: str, limit: int = 5) -> Dict:
        """
        評估測試檔案
        
        Args:
            test_file: 測試檔案路徑
            db_path: 資料庫路徑
            limit: 每個查詢返回的結果數量
            
        Returns:
            評估結果
        """
        # 載入測試資料
        test_data = load_json_dataset(test_file)
        
        # 根據資料庫路徑生成集合名稱
        corpus_name = Path(db_path).stem.replace('qgpt_', '')
        collection_name = generate_collection_name(corpus_name)
        
        # 初始化評估器
        evaluator = QGpTQueryEvaluator(db_path, collection_name)
        
        results = []
        total_hit_rate = 0
        total_precision = 0
        
        print(f"🔄 評估測試檔案: {Path(test_file).name}")
        print(f"   查詢數量: {len(test_data)}")
        
        for i, item in enumerate(test_data):
            query = item.get('question', '')
            
            # 提取正確答案（根據測試檔案結構調整）
            ground_truth = []
            if 'spreadsheet_list' in item:
                ground_truth = item['spreadsheet_list']
            elif 'answer' in item:
                # 某些測試檔案可能有不同的結構
                pass
            
            # 執行評估
            eval_result = evaluator.evaluate_single_query(query, ground_truth, limit)
            results.append(eval_result)
            
            # 累計指標
            if 'hit_rate_by_id' in eval_result:
                total_hit_rate += eval_result['hit_rate_by_id']
            if 'precision_at_k' in eval_result:
                total_precision += eval_result['precision_at_k']
            
            # 顯示進度
            if (i + 1) % 10 == 0:
                print(f"   處理進度: {i + 1}/{len(test_data)}")
        
        # 計算平均指標
        avg_hit_rate = total_hit_rate / len(test_data) if test_data else 0
        avg_precision = total_precision / len(test_data) if test_data else 0
        
        return {
            'test_file': test_file,
            'db_path': db_path,
            'total_queries': len(test_data),
            'avg_hit_rate': avg_hit_rate,
            'avg_precision': avg_precision,
            'detailed_results': results
        }
    
    def run_batch_evaluation(self, limit: int = 5, save_results: bool = True) -> Dict:
        """
        執行批次評估
        
        Args:
            limit: 每個查詢返回的結果數量
            save_results: 是否儲存詳細結果
            
        Returns:
            批次評估結果
        """
        test_files = self.get_test_files()
        if not test_files:
            print("❌ 沒有找到測試檔案")
            return {}
        
        batch_results = {}
        
        print(f"🔄 開始批次評估，找到 {len(test_files)} 個測試檔案")
        
        for test_file in test_files:
            print(f"\n{'='*60}")
            
            # 尋找對應的語料庫資料庫
            db_path = self.match_corpus_to_test(test_file)
            if not db_path:
                print(f"⚠️  跳過 {Path(test_file).name}：找不到對應的語料庫資料庫")
                continue
            
            try:
                # 執行評估
                result = self.evaluate_test_file(test_file, db_path, limit)
                batch_results[Path(test_file).stem] = result
                
                print(f"✅ 完成評估: {Path(test_file).name}")
                print(f"   平均命中率: {result['avg_hit_rate']:.4f}")
                print(f"   平均精確率: {result['avg_precision']:.4f}")
                
            except Exception as e:
                print(f"❌ 評估失敗: {Path(test_file).name} - {e}")
        
        # 儲存結果
        if save_results and batch_results:
            results_file = f"batch_evaluation_results_top{limit}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 評估結果已儲存到: {results_file}")
        
        return batch_results


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='QGpT 查詢評估器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一查詢測試
  python query_evaluator.py "財務報表" --db qgpt_Table1_mimo_ch.db
  
  # 使用測試檔案評估
  python query_evaluator.py --test-file Test_Query_and_GroundTruth_Table/MiMoTable-English_test.json --db qgpt_Table1_mimo_en.db
  
  # 批次評估所有測試集
  python query_evaluator.py --batch-eval
  
  # 批次評估並指定返回結果數量
  python query_evaluator.py --batch-eval --limit 10
        """
    )
    
    parser.add_argument('query', nargs='?', help='搜索查詢字符串')
    parser.add_argument('--db', help='向量資料庫路徑')
    parser.add_argument('--collection', help='向量集合名稱（自動從資料庫名稱推導）')
    parser.add_argument('--test-file', help='測試查詢檔案路徑')
    parser.add_argument('--batch-eval', action='store_true', help='批次評估所有測試集')
    parser.add_argument('--limit', type=int, default=5, help='返回結果數量 (預設: 5)')
    parser.add_argument('--format', choices=['detailed', 'simple', 'json'], 
                       default='detailed', help='輸出格式 (預設: detailed)')
    parser.add_argument('--save', action='store_true', help='儲存評估結果到檔案')
    
    args = parser.parse_args()
    
    # 批次評估
    if args.batch_eval:
        evaluator = BatchEvaluator()
        results = evaluator.run_batch_evaluation(limit=args.limit, save_results=args.save)
        
        # 顯示總結
        if results:
            print(f"\n{'='*60}")
            print("批次評估總結:")
            for test_name, result in results.items():
                print(f"  {test_name}:")
                print(f"    查詢數量: {result['total_queries']}")
                print(f"    平均命中率: {result['avg_hit_rate']:.4f}")
                print(f"    平均精確率: {result['avg_precision']:.4f}")
        
        return
    
    # 單一查詢或測試檔案評估
    if not args.db:
        print("❌ 請提供資料庫路徑 (--db)")
        sys.exit(1)
    
    # 自動生成集合名稱
    if not args.collection:
        corpus_name = Path(args.db).stem.replace('qgpt_', '')
        collection_name = generate_collection_name(corpus_name)
    else:
        collection_name = args.collection
    
    # 初始化評估器
    evaluator = QGpTQueryEvaluator(args.db, collection_name)
    
    # 測試檔案評估
    if args.test_file:
        if not Path(args.test_file).exists():
            print(f"❌ 找不到測試檔案: {args.test_file}")
            sys.exit(1)
        
        batch_evaluator = BatchEvaluator()
        result = batch_evaluator.evaluate_test_file(args.test_file, args.db, args.limit)
        
        print(f"\n評估結果:")
        print(f"測試檔案: {result['test_file']}")
        print(f"查詢總數: {result['total_queries']}")
        print(f"平均命中率: {result['avg_hit_rate']:.4f}")
        print(f"平均精確率: {result['avg_precision']:.4f}")
        
        if args.save:
            results_file = f"evaluation_{Path(args.test_file).stem}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"詳細結果已儲存到: {results_file}")
        
        return
    
    # 單一查詢
    if args.query:
        results = evaluator.search(args.query, args.limit)
        output = format_search_results(results, args.query, args.format)
        print(output)
        return
    
    # 如果沒有提供參數，顯示幫助
    parser.print_help()


if __name__ == "__main__":
    main()
