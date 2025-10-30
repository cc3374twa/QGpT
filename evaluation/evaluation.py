from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
import jsonlines
import json

# Configuration
result_path = "./search_results/pymilvus"
test_data_path = "./test_datasets/MiMoTable-Chinese_test.json"
evaluation_output_path = "./evaluation_results/pymilvus"
top_k = [1,3,5,10]

# functions
def create_test_data(input_file):
    with open(input_file,'r') as file:
        data = json.load(file)
    
    print(f"Opening test_dataset: '{input_file}' for Recall@k evaluation...")
    
    queries = [item["query"] for item in data]
    ans_tables = [item["Answer_table"] for item in data]  # assuming 'answer' contains the correct table names
    
    print(f"queries example: {queries[0]}\nAns_table example: {ans_tables[0]}")
    print(f"Total {len(queries)} pairs loaded.\n")
    
    return queries, ans_tables

def Recall_k(input_file, ans_tables, top_ks):
    with open(input_file,'r') as file:
        test_results = json.load(file)
    
    print(f"Opening search result file: '{input_file}' for Recall@k evaluation...")
    # print(len(test_results), test_results[0])
    
    Recall_at_ks = []
    ks = top_ks
    
    for k in ks:
        total_recall = 0
        for item in test_results:
            idx = item['query_index']
            
            # Ground Truth for this query (list of correct tables)
            correct_tables = set(ans_tables[idx])
            # print(correct_tables)
            
            # Top-k retrieved results
            retrieved_tables = set(rank_item['entity']['FileName'] for rank_idx, rank_item in enumerate(item['results']) if rank_idx < k)
            # print(retrieved_tables)
            
            # Correct retrieved count
            correct_retrieved_count = len(correct_tables & retrieved_tables)
            # print(correct_retrieved_count)
            
            # Calculate Recall@k
            if len(correct_tables) > 0:
                recall = correct_retrieved_count / len(correct_tables)
            else:
                recall = 0  # divide by zero case
            
            total_recall += recall
        
        # Calculate average Recall@k
        average_recall = total_recall / len(test_results) if len(test_results) > 0 else 0
        Recall_at_ks.append(average_recall)
        
        print(f"Recall@{k} Original score: {total_recall:.2f} / {len(test_results)}")
        print(f"Recall@{k} average score: {average_recall*100 :.2f}")
    
    print(f"\naverage Recall@k is {sum(Recall_at_ks)*100/len(Recall_at_ks):.2f}")
    return Recall_at_ks

def write_json(output_file, data):
    result = {"Recall@k": data, "k": top_k, "Average": f"{(sum(data)/len(data))*100 :.2f}"}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file}")

def main():
    queries, ans_tables = create_test_data(test_data_path)
    
    input_file = f"{result_path}/mimo_ch_1k_token.json"
    Recall_at_ks = Recall_k(input_file, ans_tables, top_k)
    write_json(f"{evaluation_output_path}/mimo_ch_1k_token_Recall.json", Recall_at_ks)

if __name__ == "__main__":
    main()