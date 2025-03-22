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

logger.info(f'Weaviate hazir mi? {client.is_ready()}')

collection_name = "EmbeddingCollection"

if not client.collections.exists(collection_name):
    client.collections.create(
        collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=64,
            max_connections=16
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="recipeid", data_type=DataType.TEXT),
        ],
    )

# DataFrame'i oku ve df_dict oluştur
df = pd.read_parquet("id_text.parquet", engine="pyarrow")  # Dosya yolunu kendi yolunla değiştir
df_dict = pd.Series(df.text.values, index=df.ID).to_dict()
logger.info("DataFrame basariyla yuklendi ve sozluge donusturuldu.")

# NPZ dosyalarını işle
data_folder = "data"

for npz_file in os.listdir(data_folder):
    if npz_file.endswith(".npz"):
        file_path = os.path.join(data_folder, npz_file)
        logger.info(f"{npz_file} yukleniyor...")

        data = np.load(file_path)
        ids_chunk = data["ids"]
        embeddings_chunk = data["embeddings"]
        ids_chunk_str = [str(id_) for id_ in ids_chunk]
        
        collection = client.collections.get(collection_name)
        
        for idx, id_str in enumerate(ids_chunk_str):
            text = df_dict.get(int(id_str), None)
            
            if text is not None:
                collection.data.insert(
                    properties={"text": text, "recipeid": id_str},
                    vector=embeddings_chunk[idx],
                )

        logger.info(f"{npz_file} basariyla islendi ve Weaviate'e kaydedildi.")
        logger.info(f"Number of failed imports: {len(collection.batch.failed_objects)}")
        logger.info(f"Collection uzunlugu: {len(collection)}")

        # Hafızayı temizle
        del data, ids_chunk, embeddings_chunk, ids_chunk_str, collection
        gc.collect()
        gc.collect()
        logger.info(f"{npz_file} hafizadan temizlendi.")

client.close()
logger.info("Tum islemler tamamlandi.")
