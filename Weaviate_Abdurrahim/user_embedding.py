import weaviate
from weaviate.classes.query import Filter, GeoCoordinate, MetadataQuery, QueryReference
import json
import logging
from weaviate.classes.config import Configure, Property, DataType, VectorDistances


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

test_users = {
    1: [940, 982998, 84450, 14992],
    2: [1172381, 1111317, 293553],
    3: [453363, 940, 1111317, 84450, 14992],
    4: [982998, 453363, 453363, 1111317],
    5: [1139552, 100562, 237723]
}

user_embeddings = {}

for user_id, liked_recipe_ids in test_users.items():
    vectors = []

    for recipe_id in liked_recipe_ids:
        response = collection.query.fetch_objects(
            filters=Filter.by_property("recipeid").equal(str(recipe_id)),
            include_vector=True
        )
        
        # Her tarif için vektör al
        if response.objects:
            vectors.append(response.objects[0].vector["default"])
        else:
            print(f"Tarif ID {recipe_id} için vektör bulunamadı.")

    # Eğer vektörler varsa ortalamasını al
    if vectors:
        avg_embedding = [sum(i)/len(vectors) for i in zip(*vectors)]
        user_embeddings[user_id] = avg_embedding
    else:
        print(f"Kullanıcı {user_id} için hiç vektör bulunamadı.")
        
collection_name2 = "UserCollection"

if not client.collections.exists(collection_name2):
    client.collections.create(
        collection_name2,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=64,
            max_connections=16
        ),
        properties=[
            Property(name="userid", data_type=DataType.TEXT),
        ],
    )

user_collection = client.collections.get(collection_name2)

for user, emb in user_embeddings.items():
    if emb is not None:
        user_collection.data.insert(
            properties={"userid": str(user)},
            vector=emb,
        )
        
response = user_collection.query.fetch_objects(
    filters=Filter.by_property("userid").equal("1"),
    include_vector=True
)

query_vector = []
for o in response.objects:
    print(o.properties)
    query_vector = o.vector["default"]

response = user_collection.query.near_vector(
    limit=5,
    near_vector=query_vector, # your query vector goes here
    return_metadata=MetadataQuery(distance=True)
)

for o in response.objects:
    print(f"Recipe Content: {o.properties}")
    print(f"Cosine Distance: {o.metadata.distance}")
    print(f"Cosine Similarity: {1 - o.metadata.distance}")

client.close()