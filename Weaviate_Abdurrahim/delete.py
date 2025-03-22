import weaviate

client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)

print(client.is_ready())

collection_name = "EmbeddingCollection"
collection = client.collections.get(collection_name)
print(len(collection))
client.collections.delete_all()
client.close()