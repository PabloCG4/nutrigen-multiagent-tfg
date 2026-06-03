from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from src.text_engine.rag.retrieval_service import RecipeRetriever, get_recipe_retriever

# --- INPUT CONFIGURATION ---
class RecipeSearchInput(BaseModel):
    query: str = Field(
        description="The semantic search query for recipes (e.g., 'gluten-free pasta', 'pollo al horno')."
    )

# --- THE AGENT TOOL ---
class RecipeSearchTool(BaseTool):
    """
    Tool that allows the Agent to search for recipes in our database.
    """
    name: str = "search_recipe_knowledge_base"
    description: str = (
        "Use this tool to find recipes, ingredients, and cooking instructions. "
        "Useful when the user asks for a specific dish or cooking ideas. "
    )
    args_schema: Type[BaseModel] = RecipeSearchInput
    
    # Internal variable to hold the database connection.
    # We use 'PrivateAttr' to protect it from Pydantic validation.
    search_engine: RecipeRetriever = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("--- Initializing Recipe Search Tool ---")
        self.search_engine = get_recipe_retriever()

    def _run(self, query: str) -> str:
        """
        This is the main function the Agent executes.
        """
        try:
            # SEARCH: We ask the engine for the top 5 recipes
            results = self.search_engine.search_recipes(query, top_k=5)
            
            # FORMATTING: We convert the data into a readable message
            if not results:
                return "No relevant recipes found in the database for this query."

            # We build the response string directly here (easier to read)
            message = "Here are the retrieved recipes:\n\n"
            
            for i, recipe in enumerate(results, 1):
                message += f"--- Recipe {i} ---\n"
                message += f"Title: {recipe['title']}\n"
                message += f"Source: {recipe['source']}\n"
                message += f"Link: {recipe['url']}\n"
                message += f"Ingredients: {recipe['ingredients'][:300]}...\n\n"
            
            return message
            
        except Exception as e:
            return f"Error occurred during recipe search: {str(e)}"

    async def _arun(self, query: str) -> str:
        """
        Required for async compatibility. 
        Just redirects to the main _run method.
        """
        return self._run(query)

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create the tool
    tool = RecipeSearchTool()
    
    # Simulate a search
    query = "pollo, arroz y tomate"
    print(f"\nSearching for: '{query}'...")
    
    # Execute (The Agent calls 'invoke', which internally calls '_run')
    response = tool.invoke(query)
    
    print("AGENT READABLE OUTPUT:")
    print(response)