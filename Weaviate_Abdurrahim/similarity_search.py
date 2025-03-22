import weaviate
from weaviate.classes.query import Filter, GeoCoordinate, MetadataQuery, QueryReference
import json
import logging


# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

client = weaviate.connect_to_local(
    host="127.0.0.1",
    port=8080,
    grpc_port=50051,
)

logger.info(f'Weaviate hazir mi? {client.is_ready()}')

collection_name = "EmbeddingCollection"
collection = client.collections.get(collection_name)
print(f"This collection has {len(collection)} recipes.")

"""for item in collection.iterator():
    print(item.uuid, item.properties)"""

response = collection.query.fetch_objects(
    filters=Filter.by_property("recipeid").equal("188535"),
    include_vector=True
)

query_vector = []
for o in response.objects:
    print(o.properties)
    query_vector = o.vector["default"]

response = collection.query.near_vector(
    limit=5,
    near_vector=query_vector, # your query vector goes here
    return_metadata=MetadataQuery(distance=True)
)

for o in response.objects:
    print(f"Recipe Content: {o.properties}")
    print(f"Cosine Distance: {o.metadata.distance}")
    print(f"Cosine Similarity: {1 - o.metadata.distance}")

client.close()
