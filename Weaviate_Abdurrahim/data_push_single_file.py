import numpy as np
import pandas as pd
import weaviate
from weaviate.util import generate_uuid5
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import uuid
import gc
import logging
import os

# Logger yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Weaviate istemcisi ve collection bağlanması
client = weaviate.connect_to_local(
    host="127.0.0.1",
    port=8080,
    grpc_port=50051,
)

logger.info(f'Weaviate hazır mı? {client.is_ready()}')

collection_name = "EmbeddingCollection2"

"""client.collections.create(
    collection_name,
    vectorizer_config=[
        # User-provided embeddings
        Configure.NamedVectors.none(
            name="recipe_embedding2",
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
        ),
    ],
    properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="recipeid", data_type=DataType.TEXT),
    ],
)"""

collection = client.collections.get(collection_name)
logger.info(f"Collection uzunluğu: {len(collection)}")

# DataFrame'i oku ve df_dict oluştur
df = pd.read_parquet("id_text.parquet", engine="pyarrow")  # Dosya yolunu kendi yolunla değiştir
df_dict = pd.Series(df.text.values, index=df.ID).to_dict()
logger.info("DataFrame başarıyla yüklendi ve sözlüğe dönüştürüldü.")

# NPZ dosyalarını işle
data_folder = "data"
batch_size = 8

npz_file = "data\chunk_20.npz"
file_path = "data\chunk_20.npz"

logger.info(f"{npz_file} yükleniyor...")

data = np.load(file_path)
ids_chunk = data["ids"]
embeddings_chunk = data["embeddings"]
ids_chunk_str = [str(id_) for id_ in ids_chunk]

with collection.batch.fixed_size(batch_size=batch_size) as batch:
    for idx, id_str in enumerate(ids_chunk_str):
        uuid = generate_uuid5(id_str)
        
        # if not collection.data.exists(uuid=uuid):
        text = df_dict.get(int(id_str), None)
        if text is not None:
            batch.add_object(
                properties={"text": text, "recipeid": id_str},
                uuid=uuid,
                vector=embeddings_chunk[idx],
            )

logger.info(f"{npz_file} başarıyla işlendi ve Weaviate'e kaydedildi.")
logger.info(f"Number of failed imports: {len(collection.batch.failed_objects)}")
logger.info(f"Collection uzunluğu: {len(collection)}")

# Hafızayı temizle
del data, ids_chunk, embeddings_chunk, ids_chunk_str
gc.collect()
logger.info(f"{npz_file} hafızadan temizlendi.")

logger.info("Tüm işlemler tamamlandı.")
