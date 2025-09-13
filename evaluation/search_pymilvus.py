from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
import jsonlines
import json

# Configuration
model_name = "BAAI/bge-m3"
db_name = "QGPT"
db_path = f"../embedding_db/pymilvus/{db_name}.db"
base_path = "../../Corpora"
top_k = 10

def search_queries_in_collection(embedding_model, client, collection_name, queries, top_k=10, output_file="./search_example.json"):
    # convert queries to vectors
    query_vector = embedding_model.encode_queries(queries) # bge
    query_vector = query_vector['dense'] # bge
    
    # search
    results = client.search(
        collection_name = collection_name,
        data = query_vector,  # must be a list of vectors
        output_fields = ['FileName', 'SheetName'],
        limit= top_k  # return top_k results
    )
    print(len(results))

    # format results
    formatted_results = []

    for idx, query in enumerate(queries):  # iterate over each query
        query_entry = {
            "query_index": idx,  
            "query": query,
            "results": results[idx] 
        }
        formatted_results.append(query_entry)

    # save results as JSON
    formatted_json = json.dumps(formatted_results, ensure_ascii=False, indent=4)

    # write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_json)
    
    print(f"Search results of '{collection_name}' is saved to {output_file}")

def main():
    # Initialize Milvus client 
    client = MilvusClient(db_path)
    collections = client.list_collections()
    print(f"\nCollections in DataBase '{db_name}':\n{collections}\n")
    
    # load embedding model from milvus
    embedding_model = BGEM3EmbeddingFunction(
        model_name = model_name, # Specify the model name
        device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    )

    # search in collection
    collection_name = "mimo_ch_1k_token"
    output_file = "./search_results/pymilvus/mimo_ch_1k_token.json"
    
    if collection_name in collections:  
        with open('/user_data/QGpT/evaluation/test_datasets/MiMoTable-Chinese_test.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        queries = [item["query"] for item in data]
        search_queries_in_collection(embedding_model, client, collection_name, queries, top_k, output_file)
    else:
        print(f"Collection {collection_name} does not exist in the database.")

if __name__ == "__main__":
    main()