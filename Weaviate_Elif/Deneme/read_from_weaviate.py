import weaviate

# Connect to the local Weaviate instance
client = weaviate.connect_to_local()

# Define the collection name
class_name = "Recipe"

# Get the collection
collection = client.collections.get(class_name)

# Fetch all objects from the collection
result = collection.query.fetch_objects(limit=100)  # Adjust limit as needed

# Display fetched recipes
if result.objects:
    for obj in result.objects:
        print("Name:", obj.properties.get("name", "N/A"))
        print("Ingredients:", obj.properties.get("ingredients", "N/A"))
        print("Instructions:", obj.properties.get("instructions", "N/A"))
        print("Label:", obj.properties.get("label", "N/A"))
        print("Category:", obj.properties.get("category", "N/A"))
        print("-" * 50)
else:
    print("No recipes found in Weaviate.")

# Close the client connection
client.close()
