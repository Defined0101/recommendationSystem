import logging
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    PointStruct,
)
from functools import lru_cache
from config import Config
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeSearchEngine:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = self._init_qdrant()
        self.model, self.tokenizer = self._init_model()
        self.recipes_df = pd.read_parquet(config.RECIPES_FILE)

    def _init_qdrant(self) -> QdrantClient:
        return QdrantClient(
            self.config.QDRANT_HOST,
            port=self.config.QDRANT_PORT,
            timeout=300,
            prefer_grpc=True,
        )

    def _init_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        model = AutoModel.from_pretrained(self.config.MODEL_NAME).to(self.device)
        model.eval()
        return model, tokenizer

    @torch.no_grad()
    @lru_cache(maxsize=Config.CACHE_SIZE)
    def get_embedding(self, text: str) -> np.ndarray:
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, -1, :]
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _format_ingredients(self, ingredients_str: str) -> List[dict]:
        """Format ingredients from string to structured list"""
        try:
            ingredients = ast.literal_eval(ingredients_str)
            formatted = []
            for ingredient in ingredients:
                quantity = ingredient.get("quantity", "")
                unit = ingredient.get("unit", "")
                name = ingredient.get("name", "")

                if quantity and unit:
                    formatted.append(f"{name}: {quantity} {unit}")
                else:
                    formatted.append(name)

            return formatted
        except Exception as e:
            logger.error(f"Error formatting ingredients: {e}")
            return []

    def search_recipes(
        self,
        query: str,
        query_type: str = "text",
        ingredients: Optional[List[str]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        upper_threshold: Optional[float] = None,
    ) -> List[dict]:
        try:
            qdrant_filter = (
                self._create_ingredient_filter(ingredients) if ingredients else None
            )

            if query_type == "text":
                query_vector = self.get_embedding(query)
                query_vector = query_vector.tolist()
            elif query_type == "id":
                query_vector = self.get_recipe_embedding(query)
                if query_vector is None:
                    raise ValueError(f"No embedding found for recipe ID {query}")
            else:
                raise ValueError("Invalid query_type. Use 'text' or 'id'")

            search_response = self.client.query_points(
                collection_name="text_embeddings",
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
            )

            return self._process_search_results(
                search_response.points, similarity_threshold, upper_threshold
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _create_ingredient_filter(self, ingredients: List[str]) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="ingredients", match=MatchValue(value=ing))
                for ing in ingredients
            ]
        )

    def _process_search_results(
        self,
        points: List,
        similarity_threshold: float,
        upper_threshold: Optional[float],
    ) -> List[dict]:
        results = []
        for point in points:
            if (
                similarity_threshold is not None
                and point.score < similarity_threshold
                or upper_threshold is not None
                and point.score > upper_threshold
            ):
                continue

            recipe = self.recipes_df[self.recipes_df["ID"] == point.id].iloc[0]
            results.append(
                {
                    "name": recipe["Name"],
                    "category": recipe["Category"],
                    "score": point.score,
                    "ingredients": self._format_ingredients(recipe["Ingredients"]),
                }
            )
        return results

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.client.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def get_recipe_embedding(self, recipe_id: int) -> Optional[List[float]]:
        """
        Retrieves embedding vector for given recipe ID from Qdrant.
        Returns None if recipe or vector not found.
        """
        try:
            point = self.client.retrieve(
                collection_name="text_embeddings",
                ids=[recipe_id],
                with_vectors=True
            )
            if point and point[0].vector:
                return point[0].vector
            return None
        except Exception as e:
            logger.error(f"Error retrieving recipe embedding: {e}")
            return None
