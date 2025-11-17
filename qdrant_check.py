from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle
from base_rag2 import embedd
import numpy as np

class qdrant:
    def __init_(self):
        pass
    def to_db(vec_query):
        # Step 1: Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)  # Adjust host/port if necessary
        print("Connected to Qdrant.")

        # Step 2: Define a Collection Schema
        collection_name = "db1_collection"
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Set vector size and similarity metric
        )
        print(f"Collection '{collection_name}' created successfully.")

        # Step 3: Insert Embeddings into the Collection
        # Load embeddings from the pickle file
        with open("data_dl.pkl", "rb") as f:
            print("loading the pickle")
            dic = pickle.load(f)
            embeddings = dic['embeddings']

            print(len(embeddings), len(embeddings[0]))
            print("loaded the pickle")

        # Prepare embeddings as a list of PointStruct objects
        points = [
            PointStruct(id=i,  vector=embedding) for i, embedding in enumerate(embeddings)
        ]

        # Insert embeddings into the collection
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print("Embeddings inserted successfully!")

        # Step 4: Check the Number of Points in the Collection
        count = client.count(collection_name=collection_name).count
        print(f"Number of points in collection '{collection_name}': {count}")

        # Step 5: Search for Similar Embeddings
        print(vec_query.shape)
        vec_query = vec_query.tolist()  # Convert NumPy array to Python list of floats
        #print("vector_qyery", vec_query)
        results = client.search(
            collection_name=collection_name,
            query_vector=vec_query,  # The vector you want to search for
            limit=2  # Number of nearest neighbors to retrieve
        )

        print("Search results:")
        for result in results:
            print(f"ID: {result.id}, Distance: {result.score}")

# Generate embedding for a query vector
em = embedd()
vec_query = em.generate_embeddings("the transformer architecture")

# Call the function
qdrant.to_db(vec_query.mean(axis = 0).squeeze())


"""
class QdrantDB:
    def __init__(self, host="localhost", port=6333, collection_name="db1_collection", vector_size=768):
        """
        #Initialize the QdrantDB class.

        #Args:
        #    host (str): Host address of the Qdrant server.
        #    port (int): Port of the Qdrant server.
        #    collection_name (str): Name of the collection to create or use.
        #    vector_size (int): Size of the vectors to be stored in the collection.
"""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Step 1: Connect to Qdrant
        self.client = QdrantClient(host=self.host, port=self.port)
        print("Connected to Qdrant.")

        # Step 2: Create or recreate the collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        print(f"Collection '{self.collection_name}' created successfully.")

    def to_db(self, embeddings):
        """
        #Insert embeddings into the Qdrant collection.

        #Args:
           # embeddings (list or np.ndarray): List of embeddings to insert into the collection.
"""
        # Step 3: Insert embeddings into the collection
        points = [
            PointStruct(id=i, vector=embedding) for i, embedding in enumerate(embeddings)
        ]

        self.client.upsert(
            collection_name = self.collection_name,
            points = points
        )

        print("Embeddings inserted successfully!")
        # Step 4: Check the number of points in the collection
        count = self.client.count(collection_name = self.collection_name).count
        print(f"Number of points in collection '{self.collection_name}': {count}")

    def search(self, query_vector, limit=5):
        """
        #Search for similar embeddings in the Qdrant collection.

        #Args:
        #    query_vector (list or np.ndarray): The query vector to search for.
        #    limit (int): Number of nearest neighbors to retrieve.

        #Returns:
        #    list: List of search results.
"""
        # Convert query vector to a list of floats if it's a NumPy array
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Step 5: Search for similar embeddings
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        print("Search results:")
        for result in results:
            print(f"ID: {result.id}, Distance: {result.score}")
    
        return [res.id for res in results]


    # Example usage
    def execute(self, em, query):
        # Step 1: Initialize the QdrantDB class
        qdrant_db = QdrantDB(host="localhost", port=6333, collection_name="db1_collection", vector_size=768)

        # Step 2: Load embeddings from the pickle file
        with open("data_dl.pkl", "rb") as f:
            print("Loading the pickle...")
            dic = pickle.load(f)
            chunks = dic['chunks']
            embeddings = dic['embeddings']
            print(f"Loaded {len(embeddings)} embeddings, each of size {len(embeddings[0])}.")

        # Step 3: Insert embeddings into the Qdrant collection
        qdrant_db.to_db(embeddings)

        # Step 4: Generate a query embedding
        #em = embedd()
        query_text = "the transformer architecture"
        vec_query = em.generate_embeddings(query_text).mean(axis=0).squeeze()

        # Step 5: Search for similar embeddings
        qdrant_db.search(vec_query, limit=2)
"""