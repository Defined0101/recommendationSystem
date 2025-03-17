import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import ast
import torch.nn.functional as F
import atexit
from functools import lru_cache
import signal



# Qdrant connection
client = QdrantClient("localhost", port=6333, timeout=300, prefer_grpc=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and tokenizer globally and move to device
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-en-icl")
model = AutoModel.from_pretrained("BAAI/bge-en-icl").to(device)
model.eval()  # Set model to evaluation mode

# Load recipe data
recipes_df = pd.read_parquet("recipes.parquet", engine="pyarrow")
print("Tarif kolonları:", recipes_df.columns)

def cleanup():
    """
    Properly cleans up resources when the program terminates.
    Handles model, tokenizer, and client cleanup.
    """
    try:
        # Clean up model and GPU memory
        model.cpu()
        torch.cuda.empty_cache()
        
        # Close Qdrant client connections
        if hasattr(client, '_channel'):
            client._channel.close()
        if hasattr(client, 'grpc_channel'):
            client.grpc_channel.close()
        client.close()
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Cleanup fonksiyonunu program çıkışına ve SIGINT (Ctrl+C) sinyaline kaydet
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: cleanup())




# Text embedding fonksiyonunu optimize edelim
@torch.no_grad()  # Context manager yerine decorator kullanalım
def get_text_embedding(text):
    """
    Converts given text to embedding vector.
    Returns normalized embedding vector as numpy array.
    """
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512  
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, -1, :]
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze().cpu().numpy()

# Sık aranan metinler için önbellek ekleyelim
@lru_cache(maxsize=1000)  # Son 1000 sorguyu önbellekte tutalım
def cached_get_text_embedding(text):
    """
    Caches and returns text embeddings for frequently searched queries.
    Cache size: 1000 most recent queries
    """
    return get_text_embedding(text)

def get_recipe_id(query_text):
    """
    Kullanıcının girdiği query_text'i en yakın tarifin id'sini döndürür.
    """
    # Önce query_text'e en benzer yemek ismini bulalım
    query_vector = get_text_embedding(query_text)
    
    search_response = client.query_points(
        collection_name="text_embeddings",
        query=query_vector.tolist(),
        limit=1,  # En yakın tek tarifi al
        with_payload=True
    )
    
    # Eğer en yakın tarif bulunamazsa None döndür
    if not search_response.points:
        return None

    # En yakın tarifin ID'sini döndür
    return search_response.points[0].id


from qdrant_client.models import Filter, FieldCondition, MatchValue
import json

def search_recipes(query=None, query_type="text", ingredients=None, limit=10, similarity_threshold=0.0, upper_threshold=None):
    """
    Searches for similar recipes based on query and filters by ingredients.
    
    Args:
        query: Search text or recipe ID
        query_type: Query type ("text" or "id")
        ingredients: List of ingredients to filter by
        limit: Maximum number of recipes to return
        similarity_threshold: Minimum similarity score
        upper_threshold: Maximum similarity score
    """
    try:
        if query is None:
            print("Lütfen bir sorgu (query) belirtin.")
            return
        
        recipe_id = None

        # Kullanıcının belirttiği malzemelere göre Qdrant Filter nesnesi oluştur
        qdrant_filter = None
        if ingredients:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="ingredients",
                        match=MatchValue(value=ing)  # `MatchValue` ile her malzemeyi eşleştir
                    ) for ing in ingredients
                ]
            )

        if query_type == "text":
            # Önbellekli embedding fonksiyonunu kullan
            query_vector = cached_get_text_embedding(query)

            search_response = client.query_points(
                collection_name="text_embeddings",
                query=query_vector.tolist(),
                query_filter=qdrant_filter,  # ✅ Qdrant `Filter` nesnesi kullanılıyor
                limit=limit,
                with_payload=True
            )

        elif query_type == "id":
            recipe_id = query
            print(f"\nArama Sonuçları: ID {query} ile benzer tarifler")

            # ID ile benzer tarifleri bulurken de malzeme filtresi uygula
            search_response = client.query_points(
                collection_name="text_embeddings",
                query=get_recipe_embedding(recipe_id),  # ID'den embedding al
                query_filter=qdrant_filter,  # ✅ Qdrant `Filter` nesnesi kullanılıyor
                limit=limit,
                with_payload=True
            )
        
        else:
            print("Geçersiz sorgu tipi. 'text' veya 'id' kullanın.")
            return
        
        # Gelen sonuçları işle
        hits = search_response if isinstance(search_response, list) else search_response.points

        print("\n✅ Önerilen Tarifler:")
        print("-" * 50)

        for point in hits:
            if similarity_threshold is not None and point.score < similarity_threshold:
                continue
            if upper_threshold is not None and point.score > upper_threshold:
                continue
            
            try:
                matching_recipes = recipes_df[recipes_df['ID'] == point.id]
                if not matching_recipes.empty:
                    recipe = matching_recipes.iloc[0]
                    # Malzemeleri düzgün bir şekilde formatla

                    print(f"Tarif Adı: {recipe['Name']}")
                    print(f"Kategori: {recipe['Category']}")
                    print(f"Benzerlik Skoru: {point.score:.4f}")
                    
                    # Malzemeleri düzgün formatta yazdır
                    print("Malzemeler:")
                    ingredients = ast.literal_eval(recipe['Ingredients'])
                    for ingredient in ingredients:
                        quantity = ingredient.get('quantity', '')
                        unit = ingredient.get('unit', '')
                        name = ingredient.get('name', '')
                        
                        # Miktar ve birim varsa yazdır, yoksa sadece malzeme adını yazdır
                        if quantity and unit:
                            print(f"  • {name}: {quantity} {unit}")
                        else:
                            print(f"  • {name}")
                            
                    print("-" * 30)
                else:
                    print(f"Tarif bulunamadı (ID: {point.id})")
            except Exception as e:
                print(f"Sonuç işleme hatası: {str(e)}")
                
    except Exception as e:
        print(f"Arama sırasında bir hata oluştu: {str(e)}")


# Test için kullanıcıların beğendiği tarifleri simüle edelim
test_user_interactions = {
    101: [5, 13, 27],  # Kullanıcı 101 -> Tarif ID'leri 5, 12, 27'yi beğenmiş
    102: [9, 13, 19],  # Kullanıcı 102 -> Tarif ID'leri 8, 12, 19'u beğenmiş
    103: [4, 9, 27],   # Kullanıcı 103 -> Tarif ID'leri 3, 8, 27'yi beğenmiş
    104: [5, 27]    # Kullanıcı 104 -> Tarif ID'leri 1, 2, 3'ü beğenmiş
}

# Kullanıcıların embeddinglerini test için hesaplayalım
test_user_embeddings = {}

def get_recipe_embedding(recipe_id):
    """
    Retrieves embedding vector for given recipe ID from Qdrant.
    Returns None if recipe or vector not found.
    """
    try:
        point = client.retrieve(
            collection_name="text_embeddings",
            ids=[recipe_id],
            with_vectors=True
        )
        if point and point[0].vector:
            return point[0].vector
    except Exception as e:
        print(f"Error retrieving recipe embedding: {e}")
    return None

def calculate_user_embeddings():
    """
    Calculates embeddings for test users based on their recipe interactions.
    Returns dictionary of user embeddings.
    """
    print("\nKullanıcı embedding'leri hesaplanıyor...")
    calculated_embeddings = {}
    
    for user_id, recipe_ids in test_user_interactions.items():
        recipe_embeddings = []
        print(f"\nKullanıcı {user_id} için tarif embedding'leri alınıyor:")
        
        for recipe_id in recipe_ids:
            recipe_vector = get_recipe_embedding(recipe_id)
            if recipe_vector is not None:
                recipe_embeddings.append(recipe_vector)
                print(f"Tarif ID {recipe_id} embedding'i alındı")
        
        if recipe_embeddings:
            calculated_embeddings[user_id] = torch.tensor(recipe_embeddings).mean(dim=0).numpy()
            print(f"Kullanıcı {user_id} embedding'i oluşturuldu")
    
    return calculated_embeddings

def store_test_user_embeddings():
    """
    Stores test user embeddings in Qdrant.
    Creates collection if it doesn't exist.
    """
    try:
        # Önce koleksiyonun var olup olmadığını kontrol et
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        # Eğer koleksiyon yoksa oluştur
        if "test_user_embeddings" not in collection_names:
            client.create_collection(
                collection_name="test_user_embeddings",
                vectors_config=VectorParams(
                    size=4096,
                    distance="Cosine"
                )
            )
            print("'test_user_embeddings' koleksiyonu oluşturuldu.")
        
        # PointStruct kullanarak points listesi oluştur
        test_embeddings = [
            PointStruct(
                id=int(user_id),
                vector=embedding.tolist(),
                payload={"user_id": int(user_id)}
            )
            for user_id, embedding in test_user_embeddings.items()
        ]

        if test_embeddings:  # Boş liste kontrolü
            client.upsert(
                collection_name="test_user_embeddings",
                points=test_embeddings,
                wait=True
            )
            print(f"Test kullanıcı embedding'leri yüklendi: {len(test_embeddings)} kullanıcı.")
        else:
            print("Yüklenecek embedding bulunamadı.")
            
    except Exception as e:
        print(f"Test kullanıcı embedding'leri yüklenirken hata oluştu: {e}")

def find_similar_users(test_user_id, top_n=3):
    """
    Finds most similar users to given test user.
    Returns list of (user_id, similarity_score) tuples.
    """
    if test_user_id not in test_user_embeddings:
        print(f"Kullanıcı {test_user_id} için embedding bulunamadı.")
        return []

    user_embedding = test_user_embeddings[test_user_id]

    search_response = client.query_points(
        collection_name="test_user_embeddings",
        query=user_embedding.tolist(),
        limit=top_n,
        with_payload=True,
        score_threshold=0.0  # Minimum benzerlik skoru
    )

    similar_users = [(hit.payload["user_id"], hit.score) 
                    for hit in search_response.points 
                    if hit.payload["user_id"] != test_user_id]
                    
    print(f"\nKullanıcı {test_user_id} için en benzer kullanıcılar:")
    for user_id, score in similar_users:
        print(f"Kullanıcı {user_id}: Benzerlik Skoru = {score:.4f}")
        
    return similar_users

def recommend_test_user_recipes(user_id, limit=3):
    """
    Recommends recipes based on similar users' preferences.
    Uses similarity scores as weights for recommendations.
    """
    similar_users = find_similar_users(user_id)

    if not similar_users:
        print(f"Kullanıcı {user_id} için öneri yapılamadı.")
        return []

    # Benzerlik skorlarını kullanarak ağırlıklı öneriler oluştur
    recipe_scores = {}
    for similar_user_id, similarity_score in similar_users:
        user_recipes = test_user_interactions.get(similar_user_id, [])
        for recipe_id in user_recipes:
            if recipe_id not in recipe_scores:
                recipe_scores[recipe_id] = 0
            recipe_scores[recipe_id] += similarity_score  # Benzerlik skorunu ağırlık olarak kullan

    # Tarifleri toplam skorlarına göre sırala
    sorted_recipes = sorted(recipe_scores.items(), key=lambda x: x[1], reverse=True)
    top_recipes = [recipe_id for recipe_id, score in sorted_recipes[:limit]]

    print(f"\nKullanıcı {user_id} için önerilen tarifler:")
    for recipe_id, score in sorted_recipes[:limit]:
        recipe_name = recipes_df[recipes_df['ID'] == recipe_id]['Name'].iloc[0]
        print(f"Tarif ID {recipe_id} ({recipe_name}): Ağırlıklı Skor = {score:.4f}")
    
    return top_recipes

if __name__ == "__main__":
    # Text bazlı arama örneği
    print("\nBasic text search example:")
    search_recipes(query="Chicken Salad", query_type="text", limit=5)
    print("-" * 50)

    # Malzeme filtreli arama örneği
    print("\nSearch with ingredient filter:")
    search_recipes(
        query="Chicken Salad", 
        query_type="text", 
        ingredients=["avocado", "lettuce"], 
        limit=5
    )
    print("-" * 50)
    
    # ID ve malzeme filtreli arama örneği
    print("\nSearch by ID with ingredient filter:")
    search_recipes(
        query=5, 
        query_type="id", 
        ingredients=["tomato", "butter"], 
        limit=5
    )
    print("-" * 50)
    
    
    