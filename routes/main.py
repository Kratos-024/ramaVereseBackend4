# from fastapi import FastAPI
# from pydantic import BaseModel
# from model.main import get_answer
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# class VerseRequest(BaseModel):
#     verse: str
    
    
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"], # Allows all methods
#     allow_headers=["*"], # Allows all headers
# )
# @app.get("/")
# def read_root():
#     return {"message": "Hello world"}

# @app.post("/get-verse")
# def get_verse(request: VerseRequest):
#     try:
#         response = get_answer(request.verse) 
#         return response
#     except Exception as e:
#         return {"error": str(e)}







from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
from contextlib import asynccontextmanager
import asyncio
import gc


from model.main import get_answer, initialize_system

# Request/Response models
class VerseRequest(BaseModel):
    query: str

class VerseResponse(BaseModel):
    answer: str
    query: str = ""
    error: str = None

# Global initialization flag
_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global _initialized
    
    # Startup
    print("🚀 Starting FastAPI application...")
    try:
        if not _initialized:
            print("Initializing QA system...")
            # Initialize in background to avoid blocking startup
            asyncio.create_task(initialize_in_background())
        else:
            print("✅ QA system already initialized")
    except Exception as e:
        print(f"❌ Failed to start initialization: {str(e)}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down FastAPI application...")
    gc.collect()

async def initialize_in_background():
    """Initialize system in background"""
    global _initialized
    try:
        await asyncio.get_event_loop().run_in_executor(None, initialize_system)
        _initialized = True
        print("✅ QA system initialized successfully")
    except Exception as e:
        print(f"❌ Background initialization failed: {str(e)}")

# Create FastAPI app (minimal)
app = FastAPI(
    title="Verse QA API",
    description="Simple API for verse queries",
    version="1.0.0",
    lifespan=lifespan
)

# Minimal CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/helloworld")
async def hello_world():
    """Simple test endpoint"""
    return {"message": "Hello World! API is running."}

@app.post("/get-verse", response_model=VerseResponse)
async def get_verse(request: VerseRequest):
    """Get verse answer based on query"""
    global _initialized
    
    if not _initialized:
        raise HTTPException(
            status_code=503, 
            detail="System still initializing. Please try again in a moment."
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Process query
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_answer, request.query)
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {result['error']}"
            )
        
        return VerseResponse(
            answer=result.get("answer", "No answer generated"),
            query=result.get("query", request.query)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# Minimal error handler
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found"}

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return {"error": "Internal server error"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,  # Direct app reference
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="warning"  # Reduce logging
    )