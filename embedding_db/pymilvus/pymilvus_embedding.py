from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import jsonlines
import json

# Configuration
model_name = "BAAI/bge-m3"
db_name = "QGPT"
db_path = f"./{db_name}.db"
base_path = "../../Corpora"
batch_size = 1000

# functions
def Loading_embedding_model(model_name): # loading embedding model
    print(f"Loading tokenizer for model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_token = tokenizer.model_max_length
    print(f"{max_token} is the max length of the tokenizer.")
    
    print(f"Start loading embedding model '{model_name}'...") 
    embedding_model = BGEM3FlagModel(model_name,  use_fp16=True, devices="cuda") # Setting use_fp16 to True speeds up computation with a slight performance degradation
    print(f"'{model_name}' Model loaded.")
    
    return embedding_model, tokenizer, max_token 

def embedding_texts(texts, embedding_model, token_limit, batch_size=1000): # embedding in batches
    embeddings = []

    # process data in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding documents", unit="batch"):
        # extract current batch data
        batch_docs = texts[i:i + batch_size]
        
        # generate vectors in batch
        encode_batch = embedding_model.encode(batch_docs, max_length=token_limit)['dense_vecs'] # bge encoder
        
        # convert to float32 and reshape
        batch_reform = [np.asarray(vec, dtype=np.float32) for vec in encode_batch]
        
        # Store in list
        embeddings.extend(batch_reform)

    print(f"Total {len(texts)} texts to be embedded.")
    return embeddings

def insert_data(client, collection_name, embeddings, texts, ids, file_names, sheet_names, batch_size=1000): # insert data into collection
    # batch insert data
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Inserting data"):
        start_idx = i
        end_idx = min(i + batch_size, len(embeddings))
        
        # prepare batch data
        batch_data = [
            {
                "id": ids[j],
                "SheetName": sheet_names[j],  
                "vector": embeddings[j],  # bge dense vector
                "FileName": file_names[j],  
                "Text": texts[j][:8192],  # text
            }
            for j in range(start_idx, end_idx)
        ]

        # insert the batch data into the collection
        result = client.insert(collection_name=collection_name, data=batch_data)

        # print result
        print(f"Inserted {len(result)} records into collection '{collection_name}'.")

    print("Data insertion completed.")

def main(): 
    # Initialize Milvus client 
    client = MilvusClient(db_path)
    collections = client.list_collections()
    print(f"\nCollections in DataBase '{db_name}':\n{collections}\n")
    # load embedding model
    embedding_model, tokenizer, token_limit = Loading_embedding_model(model_name)
    
    # test for a file embedding to db
    with open('/user_data/QGpT/Corpora/Table1_mimo_table_length_variation/mimo_ch/1k_token.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ids, texts, file_names, sheet_names = (
        [item["id"] for item in data],
        [item["Text"] for item in data],
        [item["FileName"] for item in data],
        [item["SheetName"] for item in data]
    )
    
    print(f"First data's id:{ids[0]}\n")
    print(f"First data's file name:{file_names[0]}\n")
    print(f"First data's sheet name:{sheet_names[0]}\n")
    print(f"First data's text (length {len(texts[0])}):\n{texts[0][:100]}...\n")

    embeddings = embedding_texts(texts, embedding_model, token_limit, batch_size)
    collection_name = "mimo_ch_1k_token" if "mimo_ch_1k_token" in collections else None

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"First embedding vector (dimension - {len(embeddings[0])}):\n{embeddings[0]}")
    
    if collection_name:
        insert_data(client, collection_name, embeddings, texts, ids, file_names, sheet_names, batch_size)
    else:
        print(f"Collection {collection_name} does not exist in the database.")

if __name__ == "__main__":
    main()

