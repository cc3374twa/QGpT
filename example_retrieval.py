"""
QGpT: Example Retrieval Script
示例檢索腳本，展示如何使用 QGpT 進行表格檢索

This script demonstrates the basic usage of QGpT table retrieval system
with sample queries and different output formats.
"""

import subprocess
import sys
import os

def run_qgpt_search(query: str, limit: int = 3, format_type: str = "simple"):
    """
    執行 QGpT 搜索
    
    Args:
        query: 搜索查詢
        limit: 結果數量限制
        format_type: 輸出格式 (simple, detailed, json)
    """
    try:
        cmd = [
            sys.executable, 
            "qgpt_search.py", 
            query, 
            "-n", str(limit), 
            "-f", format_type
        ]
        
        print(f"🔍 執行查詢: {query}")
        print("-" * 60)
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ 搜索失敗: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

def main():
    """
    QGpT 示例檢索主程式
    
    展示不同類型的查詢和輸出格式
    """
    print("🎯 QGpT 表格檢索系統 - 示例腳本")
    print("=" * 80)
    
    # 檢查必要檔案是否存在
    if not os.path.exists("qgpt_search.py"):
        print("❌ 找不到 qgpt_search.py，請確保在正確的目錄中執行")
        return
    
    if not os.path.exists("milvus_qgpt_tables.db"):
        print("❌ 找不到 QGpT 資料庫，請先執行 corpus_embedding_builder.py 建立資料庫")
        return
    
    # 示例查詢列表
    sample_queries = [
        # 中文查詢
        ("財務報表", "尋找財務相關的表格"),
        ("建築工程", "搜索建築和工程相關資料"),
        ("學生成績", "查找教育和學習相關表格"),
        
        # 英文查詢
        ("financial statements", "Find financial related tables"),
        ("construction project", "Search construction and engineering data"),
        ("bank interest rates", "Look for banking and interest rate information")
    ]
    
    print("📋 執行示例查詢:")
    print()
    
    for i, (query, description) in enumerate(sample_queries, 1):
        print(f"📌 示例 {i}: {description}")
        run_qgpt_search(query, limit=2, format_type="simple")
        print()
    
    print("=" * 80)
    print("✅ 示例檢索完成！")
    print()
    print("💡 使用說明:")
    print("• 直接使用: python qgpt_search.py \"您的查詢\"")
    print("• 詳細選項: python qgpt_search.py --help")
    print("• 運行示例: python example_retrieval.py")

if __name__ == "__main__":
    main()
