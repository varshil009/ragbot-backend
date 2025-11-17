from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://efc0d45f-e82c-4ba4-8d9a-f0e470a9784c.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.NIaUVZs6o3pJB6oKYm3BnR__VvodedNOSFq0bPso3L4",
)

print(qdrant_client.get_collections())