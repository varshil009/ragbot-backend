from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle
import os

client = QdrantClient(url="", 
            api_key="",
            )
def ensure_collection_exists(collection_name, vector_size):
    """Create collection if it doesn't exist."""
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        print(f"Creating collection '{collection_name}' ...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

def prepare_points(embeddings, texts, start_id):
    """Prepare Points with unique IDs."""
    return [
        PointStruct(
            id=start_id + i,
            vector=embeddings[i],
            payload={"text": texts[i]}
        )
        for i in range(len(embeddings))
    ]

def upload_in_batches(collection_name, embeddings, texts, batch_size=200):
    """Upload embeddings in safe batches to avoid timeout."""
    total = len(embeddings)
    print(f"Uploading {total} points to '{collection_name}'...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        print(f" → Uploading batch {start} to {end}...")

        batch_embeddings = embeddings[start:end]
        batch_texts = texts[start:end]

        points = prepare_points(batch_embeddings, batch_texts, start_id=start)

        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"Completed uploading all {total} points to '{collection_name}'.")


# ---------------------------------------------------
# UPLOAD DEEP LEARNING
# ---------------------------------------------------
with open("data_dl.pkl", "rb") as f:
    dl_data = pickle.load(f)
    dl_embeddings = dl_data['embeddings']
    dl_texts = dl_data['chunks']

ensure_collection_exists("data_dl_collection", len(dl_embeddings[0]))
upload_in_batches("data_dl_collection", dl_embeddings, dl_texts, batch_size=100)

# ---------------------------------------------------
# UPLOAD DATA SCIENCE
# ---------------------------------------------------
with open("data_ds.pkl", "rb") as f:
    ds_data = pickle.load(f)
    ds_embeddings = ds_data['embeddings']
    ds_texts = ds_data['chunks']

ensure_collection_exists("data_ds_collection", len(ds_embeddings[0]))
upload_in_batches("data_ds_collection", ds_embeddings, ds_texts, batch_size=100)
