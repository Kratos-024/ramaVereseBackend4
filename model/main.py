from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
from data.drive import Drive
import traceback
import gc
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
apiKey = os.getenv('ApiKey')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apiKey

# Memory optimization: Only initialize what's needed
def initialize_drive_and_files():
    """Initialize drive and download files if needed"""
    drive = Drive()
    authorized = drive.authorize()
    
    # File IDs
    faissFileid = "1CcT7gZMQrkZB6S4WalAFEb_rIZMJOHXx"
    pklFileid = "1WJpJIvpERopl9ANDuq-a-Vi6PfHrS5SB"
    
    faiss_filename = "index.faiss"
    pkl_filename = "index.pkl"
    
    # Download files if not already present
    if not os.path.isfile(faiss_filename):
        drive.download_file(authorized, faissFileid, faiss_filename)
        print(f"Downloaded {faiss_filename}")
    else:
        print(f"{faiss_filename} already exists, skipping download.")
    
    if not os.path.isfile(pkl_filename):
        drive.download_file(authorized, pklFileid, pkl_filename)
        print(f"Downloaded {pkl_filename}")
    else:
        print(f"{pkl_filename} already exists, skipping download.")
    
    return authorized

# Global variables to avoid reloading
_db = None
_qa_chain = None
_embeddings_model = None

def get_lightweight_embeddings():
    """Use a much lighter embedding model"""
    global _embeddings_model
    
    if _embeddings_model is not None:
        return _embeddings_model
    
    # Use a much smaller model - this is crucial for memory
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Much smaller than bge-small-en
    print(f"Loading lightweight embeddings model: {embedding_model}")
    
    _embeddings_model = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            'device': 'cpu',
            'trust_remote_code': False
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 1,  # Process one at a time to save memory
            'show_progress_bar': False
        }
    )
    
    return _embeddings_model

def get_qa_chain():
    """Lazy loading of QA chain with maximum memory optimization"""
    global _db, _qa_chain
    
    if _qa_chain is not None:
        return _qa_chain
    
    try:
        # Force garbage collection before loading
        gc.collect()
        
        # Load embeddings
        embeddings_model = get_lightweight_embeddings()
        
        # Load FAISS database
        print("Loading FAISS index...")
        _db = FAISS.load_local(".", embeddings_model, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded successfully")
        
        # Use a much smaller, more efficient model
        modelName = "microsoft/DialoGPT-small"  # Much smaller than zephyr-7b-beta
        print(f"Initializing lightweight LLM: {modelName}")
        
        llm = HuggingFaceEndpoint(
            repo_id=modelName,
            temperature=0.1,
            max_new_tokens=50,  # Very small output
            timeout=20,
            model_kwargs={
                "max_length": 100,
                "do_sample": False,
                "pad_token_id": 50256
            }
        )
        
        # Simplified prompt for yes/no answers
        prompt_template = """Context: {context}
Question: {question}
Answer only "Yes" or "No":"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create RetrievalQA chain with minimal retrieval
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_db.as_retriever(search_kwargs={'k': 1}),  # Only 1 document
            chain_type_kwargs={'prompt': PROMPT},
            return_source_documents=False  # Don't return sources to save memory
        )
        
        print("✅ QA Chain initialized successfully")
        
        # Aggressive garbage collection
        gc.collect()
        
        return _qa_chain
        
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        traceback.print_exc()
        # Clean up on error
        _cleanup_memory()
        raise

def _cleanup_memory():
    """Cleanup memory by deleting large objects"""
    global _db, _qa_chain, _embeddings_model
    
    try:
        if _qa_chain is not None:
            del _qa_chain
            _qa_chain = None
        
        if _db is not None:
            del _db
            _db = None
            
        if _embeddings_model is not None:
            del _embeddings_model
            _embeddings_model = None
            
        gc.collect()
        print("Memory cleanup completed")
    except Exception as e:
        print(f"Error during memory cleanup: {e}")

def clean_query(query: str) -> str:
    """Clean and sanitize user query"""
    if not query:
        return ""
    
    # Handle encoding issues
    query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")
    
    # Remove special characters but keep basic punctuation
    query = re.sub(r'[^\w\s.,?!-]', '', query)
    
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query)
    
    return query.strip()[:200]  # Shorter limit

def get_answer(query: str):
    """Main function to get answers from the QA system"""
    try:
        if not query or not query.strip():
            return {"error": "Empty query provided"}
        
        cleaned_query = clean_query(query)
        print(f"Processing query: {cleaned_query[:50]}...")
        
        # Get QA chain (lazy loaded)
        qa_chain = get_qa_chain()
        
        # Get response
        response = qa_chain.invoke({"query": cleaned_query})
        raw_answer = response.get("result", "").strip()
        
        # Clean up the answer to be just Yes/No
        if "yes" in raw_answer.lower():
            clean_answer = "Yes"
        elif "no" in raw_answer.lower():
            clean_answer = "No"
        else:
            clean_answer = "No"  # Default to No if unclear
        
        print(f"Generated answer: {clean_answer}")
        
        # Force garbage collection after processing
        gc.collect()
        
        return {
            "answer": clean_answer,
            "sources": [],  # No sources to save memory
            "query": cleaned_query
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        
        # Cleanup on error
        _cleanup_memory()
        gc.collect()
        
        return {"error": error_msg}

def initialize_system():
    """Initialize the system - call this once on startup"""
    try:
        print("Initializing system...")
        
        # Set environment variables for memory optimization
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        authorized = initialize_drive_and_files()
        print(f"Drive authorized: {authorized}")
        
        # Don't pre-load the QA chain - load on demand
        print("✅ System initialization complete (lazy loading enabled)")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        _cleanup_memory()
        raise

def health_check():
    """Simple health check for the application"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "status": "healthy",
            "qa_chain_loaded": _qa_chain is not None,
            "db_loaded": _db is not None,
            "memory_usage_mb": round(memory_mb, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    # For testing
    initialize_system()
    
    # Test query
    test_query = "Is this system working?"
    result = get_answer(test_query)
    print(f"Test result: {result}")