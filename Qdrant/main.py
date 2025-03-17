from recipe_search_engine import RecipeSearchEngine
from config import Config

def main():
    config = Config()
    engine = RecipeSearchEngine(config)
    
    try:
        results = engine.search_recipes(
            query=5,
            query_type="id",
            limit=5
        )
        
        for result in results:
            print(f"Recipe: {result['name']}")
            print(f"Score: {result['score']:.4f}")
            print("Ingredients:", result['ingredients'])
            print("-" * 50)
            
    finally:
        engine.cleanup()

if __name__ == "__main__":
    main() 