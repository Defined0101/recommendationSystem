from qdrant_client import QdrantClient

# Qdrant bağlantısını başlat
client = QdrantClient("localhost", port=6333, timeout=300, prefer_grpc=True)

# Tarif verilerini yükleyelim
import pandas as pd
recipes_df = pd.read_parquet("recipes.parquet", engine="pyarrow")


def update_payloads():
    """
    Mevcut embedding noktalarının payload'larını tarif bilgileriyle günceller.
    """
    for index, row in recipes_df.iterrows():
        recipe_id = int(row["ID"])  # Embedding'lerde bulunan recipe_id ile eşleştireceğiz
        
        payload = {
            "name": row["Name"],
            "category": row["Category"],
            "instructions": row["Instructions"],
            "ingredients": row["Ingredients"]
        }
        
        # Mevcut noktaya payload ekle
        client.set_payload(
            collection_name="text_embeddings",
            payload=payload,
            points=[recipe_id]  # Güncellenecek noktalar
        )
        
        print(f"✅ Tarif ID {recipe_id} için payload güncellendi.")

if __name__ == "__main__":
    print("Tarifler yüklendi.")
    update_payloads()
