from dataclasses import dataclass

@dataclass
class Config:
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    MODEL_NAME: str = "BAAI/bge-en-icl"
    RECIPES_FILE: str = "recipes.parquet"
    EMBEDDING_SIZE: int = 4096
    CACHE_SIZE: int = 1000 