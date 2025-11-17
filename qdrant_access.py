from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from base_rag2 import embedd

client = QdrantClient(host="localhost", port=6333)
em = embedd()
vec_query = em.generate_embeddings("the transformer architecture")
print(vec_query.shape)

vec_query = vec_query.mean(axis = 0).squeeze()
print(vec_query.shape)
 # Convert NumPy array to Python list of floats
#print("vector_qyery", vec_query)
results = client.search(
    collection_name="data_dl_collection",
    query_vector=vec_query.tolist(),  # The vector you want to search for
    limit=3  # Number of nearest neighbors to retrieve
)

print("Search results:")
for result in results:
    print(f"ID: {result.id}, Distance: {result.score}")
