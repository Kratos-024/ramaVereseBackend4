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
import psutil
import gc

# Import your main QA functions
from model.main import get_answer, health_check, initialize_system

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_sources: int = 2

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    query: str = ""
    error: str = None

class HealthResponse(BaseModel):
    status: str
    qa_chain_loaded: bool = False
    db_loaded: bool = False
    memory_usage_mb: float = 0.0
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
            await asyncio.get_event_loop().run_in_executor(None, initialize_system)
            _initialized = True
            print("✅ QA system initialized successfully")
        else:
            print("✅ QA system already initialized")
    except Exception as e:
        print(f"❌ Failed to initialize QA system: {str(e)}")
        # Don't raise here - let the app start but mark as unhealthy
    
    yield
    
    # Shutdown
    print("🛑 Shutting down FastAPI application...")
    # Force garbage collection on shutdown
    gc.collect()

# Create FastAPI app
app = FastAPI(
    title="LangChain QA System",
    description="A Question-Answering system using LangChain and FAISS",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LangChain QA System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Health check endpoint"""
    try:
        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get health status
        health = health_check()
        
        return HealthResponse(
            status=health.get("status", "unknown"),
            qa_chain_loaded=health.get("qa_chain_loaded", False),
            db_loaded=health.get("db_loaded", False),
            memory_usage_mb=round(memory_mb, 2),
            error=health.get("error")
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            error=str(e)
        )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query and return answer with sources"""
    global _initialized
    
    if not _initialized:
        raise HTTPException(
            status_code=503, 
            detail="QA system not initialized. Please try again later."
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    
    try:
        # Process query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, get_answer, request.query)
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {result['error']}"
            )
        
        # Limit sources based on request
        sources = result.get("sources", [])[:request.max_sources]
        
        return QueryResponse(
            answer=result.get("answer", "No answer generated"),
            sources=sources,
            query=result.get("query", request.query)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/memory")
async def get_memory_stats():
    """Get memory usage statistics"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "available_memory_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gc")
async def force_garbage_collection():
    """Force garbage collection (for debugging)"""
    try:
        collected = gc.collect()
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "objects_collected": collected,
            "memory_usage_mb": round(memory_mb, 2),
            "message": "Garbage collection completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "path": str(request.url)
    }

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",  # Make sure this matches your file name
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,  # Single worker to save memory
        log_level="info"
    )