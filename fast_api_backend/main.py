from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine import perform_search  # Imports the brain from our other file!

# 1. Initialize the API Server
app = FastAPI(
    title="Semantic Hotel Search API",
    description="An AI-powered B2B backend for hotel recommendations.",
    version="2.0.0" # Bumped version for new architecture!
)

# 2. Define Request Schema
class SearchQuery(BaseModel):
    query: str
    top_k: int = 12

# 3. The API Endpoint
@app.post("/search")
async def search_hotels(request: SearchQuery):
    try:
        # Offload the heavy lifting to the engine!
        engine_response = perform_search(query=request.query, top_k=request.top_k)
        
        return {
            "status": "success",
            "search_parameters": {
                "original_query": request.query,
                "filters": engine_response["filters_applied"]
            },
            "results": engine_response["matches"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))