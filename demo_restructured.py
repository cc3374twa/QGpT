# -*- coding: utf-8 -*-
"""
QGpT Demo Script
演示重整後的程式架構和功能

這個腳本展示如何使用重整後的 QGpT 程式架構：
1. 建立語料庫 embedding
2. 執行搜索查詢
3. 評估查詢效果

Author: QGpT Research Team
Repository: https://github.com/UDICatNCHU/QGpT
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """執行命令並顯示結果"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"執行命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            print("✅ 輸出:")
            print(result.stdout)
        
        if result.stderr and "UserWarning" not in result.stderr:
            print("⚠️  錯誤:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ 命令執行失敗，退出碼: {result.returncode}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return False

def main():
    """演示主程式"""
    
    # 確認在正確的目錄
    if not Path("Corpora").exists():
        print("❌ 請在 QGpT 專案根目錄執行此腳本")
        sys.exit(1)
    
    # 取得 Python 執行檔路徑
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        python_cmd = str(venv_python)
        print("✅ 使用虛擬環境中的 Python")
    else:
        python_cmd = "python"
        print("⚠️  使用系統 Python")
    
    print("🚀 QGpT 程式架構演示")
    print("本演示將展示重整後的程式架構功能")
    
    # 1. 列出所有可用語料庫
    success = run_command([
        python_cmd, 
        "corpus_embedding_builder.py", 
        "--list"
    ], "列出所有可用語料庫")
    
    if not success:
        print("❌ 無法列出語料庫，請檢查程式設定")
        return
    
    # 2. 為小型語料庫建立 embedding（如果尚未存在）
    target_corpus = "Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json"
    
    print(f"\n檢查語料庫: {target_corpus}")
    if Path(target_corpus).exists():
        run_command([
            python_cmd,
            "corpus_embedding_builder.py",
            target_corpus
        ], f"為語料庫建立 embedding: {target_corpus}")
    else:
        print(f"⚠️  找不到語料庫檔案: {target_corpus}")
    
    # 3. 列出已建立的資料庫
    run_command([
        python_cmd,
        "qgpt_search.py",
        "--list-dbs"
    ], "列出已建立的向量資料庫")
    
    # 4. 執行搜索查詢演示
    search_queries = [
        "財務報表",
        "學生資訊", 
        "銷售數據"
    ]
    
    for query in search_queries:
        # 找到第一個可用的資料庫
        db_files = list(Path('.').glob('qgpt_*.db'))
        if db_files:
            db_name = str(db_files[0])
            run_command([
                python_cmd,
                "qgpt_search.py",
                query,
                "--db", db_name,
                "--format", "simple",
                "--limit", "3"
            ], f"搜索查詢: '{query}'")
    
    # 5. 查詢評估演示
    if db_files:
        run_command([
            python_cmd,
            "query_evaluator.py",
            "銀行利率",
            "--db", str(db_files[0]),
            "--limit", "3"
        ], "查詢評估演示")
    
    print(f"\n{'='*60}")
    print("🎉 QGpT 程式架構演示完成！")
    print("\n重整後的架構特色:")
    print("✅ 清晰的職責分離（建立 embedding vs 查詢評估）")
    print("✅ 智慧的資料庫命名系統")
    print("✅ 彈性的使用介面")
    print("✅ 完整的錯誤處理")
    print("\n可用的程式:")
    print("  - corpus_embedding_builder.py: 建立語料庫 embedding")
    print("  - qgpt_search.py: 搜索介面")
    print("  - query_evaluator.py: 查詢評估")
    print("  - utils.py: 公用工具函數")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
