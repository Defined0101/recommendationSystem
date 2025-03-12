import weaviate
from weaviate.classes.config import Configure, Property, DataType
import numpy as np

# Connect to the local Weaviate instance
client = weaviate.connect_to_local()

# Define the collection schema correctly
class_name = "Recipe"

if not client.collections.exists(class_name):
    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="name", data_type=DataType.TEXT, description="Recipe Name"),
            Property(name="ingredients", data_type=DataType.TEXT, description="List of Ingredients"),
            Property(name="instructions", data_type=DataType.TEXT, description="Cooking Instructions"),
            Property(name="label", data_type=DataType.TEXT, description="Recipe Label"),
            Property(name="category", data_type=DataType.TEXT, description="Recipe Category")
        ]
    )

# Load your embedded data from NPZ file
embedded_data_path = "embedded_data.npz" 
data = np.load(embedded_data_path, allow_pickle=True)

# Extracting the data from NPZ
names = data["names"]
ingredients = data["ingredients"]
instructions = data["instructions"]
labels = data["labels"]
categories = data["categories"]
embeddings = data["embeddings"]

# Get the collection
collection = client.collections.get(class_name)

# Upload data to Weaviate
for name, ingredient_list, instruction, label, category, embedding in zip(names, ingredients, instructions, labels, categories, embeddings):
    properties = {
        "name": str(name),
        "ingredients": str(ingredient_list),
        "instructions": str(instruction),
        "label": str(label),
        "category": str(category)
    }
    
    collection.data.insert(
        #collection=class_name,
        properties=properties,
        vector=embedding.tolist()  # Convert NumPy array to list
    )

print("Recipe data uploaded successfully to Weaviate!")

client.close()  # Free up resources