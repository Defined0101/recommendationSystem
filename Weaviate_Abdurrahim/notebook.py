# %%
import weaviate
from weaviate.classes.config import Configure, Property, DataType
import numpy as np
import pyarrow.parquet as pq

# %%
# Load the Parquet file
parquet_path = "filtered_recipes.parquet"  # Update with your file path
table = pq.read_table(parquet_path)

# Convert to Pandas DataFrame
recipes_df = table.to_pandas()

# Show sample data
print(recipes_df.head())

# %%
# Load the NPZ file
npz_path = "recipe_data_gpu0.npz"  # Update with your file path
data = np.load(npz_path, allow_pickle=True)

# Extract IDs and embeddings
ids = data["ids"]
embeddings = data["embeddings"]

# Check shape
print(f"IDs shape: {ids.shape}")
print(f"Embeddings shape: {embeddings.shape}")

# %%
# Connect to the local Weaviate instance
client = weaviate.connect_to_local()

# Define the collection schema correctly
class_name = "Recipe"

# %%
from weaviate.classes.config import Configure, VectorDistances
if not client.collections.exists(class_name):
    client.collections.create(
        name=class_name,
        description="A collection of recipes with ingredients and instructions",
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.L2
        ),
        properties=[
            Property(name="name", data_type=DataType.TEXT, description="Recipe Name"),
            Property(name="ingredients", data_type=DataType.TEXT, description="List of Ingredients"),
            Property(name="instructions", data_type=DataType.TEXT, description="Cooking Instructions"),
            Property(name="label", data_type=DataType.TEXT, description="Recipe Label"),
            Property(name="category", data_type=DataType.TEXT, description="Recipe Category")
        ]
    )

# %%
# Process in batches
batch_size = 500  # Adjust based on memory capacity
num_batches = len(ids) // batch_size + 1

collection = client.collections.get("Recipe")

for batch_idx in range(15):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(ids))

    # Process only a batch at a time
    batch_ids = ids[start_idx:end_idx]
    batch_embeddings = embeddings[start_idx:end_idx]

    for i, recipe_id in enumerate(batch_ids):
        recipe_data = recipes_df[recipes_df["ID"] == recipe_id]

        if not recipe_data.empty:
            row = recipe_data.iloc[0]  # Get the first match

            # ✅ Convert Ingredients to String if List
            ingredient_list = row["Ingredients"]
            if isinstance(ingredient_list, list):
                ingredient_list = ", ".join(map(str, ingredient_list))  # Ensure all items are strings

            # ✅ Ensure all fields are converted to native Python types
            properties = {
                "name": str(row["Name"]),
                "ingredients": str(ingredient_list),
                "instructions": str(row["Instructions"]),
                "label": str(row["Label"]),
                "category": str(row["Category"])
            }

            # ✅ Convert embeddings explicitly to a list
            vector_embedding = batch_embeddings[i].tolist()

            collection.data.insert(
                properties=properties,
                vector=vector_embedding  # Convert NumPy array to Python list
            )

    print(f"✅ Processed batch {batch_idx + 1} of {num_batches}")

print("🎉 All recipes uploaded successfully!")

# %%
print("Sample IDs from dataset:", ids[:10])

# %%
# ✅ Find the index of recipe ID = 1
recipe_index = np.where(ids == 4)[0]
print(f"Index of Recipe ID 4: {recipe_index}")

# ✅ Get the corresponding embedding (convert to list for Weaviate)
if len(recipe_index) > 0:
    query_embedding = embeddings[recipe_index[0]].tolist()
else:
    print("⚠️ Recipe with ID 4 not found.")
    query_embedding = None

# %%
from weaviate.classes.query import MetadataQuery
search_results = collection.query.near_vector(
    near_vector=query_embedding,  # Use the retrieved embedding
    limit=5,  # Return top 5 similar recipes
    return_properties=["name", "ingredients", "instructions", "category", "label"],  # ✅ Corrected metadata format
    return_metadata=MetadataQuery(distance=True)
)

# ✅ Display results
print("\n Top 5 Similar Recipes to Recipe ID 4:")
for i, recipe in enumerate(search_results.objects):
    print(f"\n🔹Match {i+1}")
    print(f"Name: {recipe.properties['name']}")
    print(f"Category: {recipe.properties['category']}")
    print(f"Label: {recipe.properties['label']}")
    print(f"Instructions: {recipe.properties['instructions'][:300]}...")  # Show first 300 chars
    print(f"Ingredients: {recipe.properties['ingredients'][:5]}...")  # Show first 5 ingredients
    print(f"Distance: {recipe.metadata.distance}")


# %%
sample_vector = embeddings[0].tolist()  # Pick the first embedding
search_results = collection.query.near_vector(
    near_vector=sample_vector,
    limit=5,
    return_properties=["name", "ingredients", "instructions", "category", "label"],
    return_metadata=MetadataQuery(distance=True)
)

# ✅ Display results
print("\n Top 5 Similar Recipes to Recipe ID 0:")
for i, recipe in enumerate(search_results.objects):
    print(f"\n🔹Match {i+1}")
    print(f"Name: {recipe.properties['name']}")
    print(f"Category: {recipe.properties['category']}")
    print(f"Label: {recipe.properties['label']}")
    print(f"Instructions: {recipe.properties['instructions'][:300]}...")  # Show first 300 chars
    print(f"Ingredients: {recipe.properties['ingredients'][:5]}...")  # Show first 5 ingredients
    print(f"Distance: {recipe.metadata.distance}")

# %%
#client.collections.delete("Recipe")
#print("🚮 Collection 'Recipe' deleted successfully")

# %%
client.close()      

# %%



