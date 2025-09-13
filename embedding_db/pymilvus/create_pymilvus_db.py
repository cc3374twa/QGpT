from pymilvus import MilvusClient
import os

db_name = "QGPT"
db_path = f"./{db_name}.db"

# Check if the database file exists
if not os.path.exists(db_path):
    print(f"Database {db_name} does not exist, a new database will be created at {db_path}")
client = MilvusClient(db_path)

base_path = "../../Corpora"

# Get existing collections
existing_collections = set(client.list_collections())

# Traverse all subfolders and files
for root, dirs, files in os.walk(base_path):
    for file in files:
        # Get the last folder name
        last_folder = os.path.basename(root)
        
        # Get the file name without extension
        file_name = os.path.splitext(file)[0]

        # Use "last folder name + file name" as collection name
        collection_name = f"{last_folder}_{file_name}"

        # Skip if the collection already exists
        if collection_name in existing_collections:
            print(f"⚠️ Collection already exists, skipping: {collection_name}")
            continue

        # Create a new collection
        client.create_collection(
            collection_name=collection_name,
            dimension=1024
        )
        print(f"✅ Created collection: {collection_name}")

# Print all collections in the database
print(f"\nCollections in '{db_name}':\n{client.list_collections()}\n")
