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

i = 0
for item in collection.iterator():
    print(item.properties["recipeid"])
    if i == 100:
        break
    i = i + 1
    
client.close()