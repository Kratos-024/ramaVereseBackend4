from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings  # Fixed imports
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
from data.drive import Drive
import traceback
import gc  # For garbage collection to help with memory

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

def get_qa_chain():
    """Lazy loading of QA chain to optimize memory usage"""
    global _db, _qa_chain
    
    if _qa_chain is not None:
        return _qa_chain
    
    try:
        # Initialize embeddings with memory optimization
        embedding_model = "BAAI/bge-small-en"
        print("Loading embeddings model...")
        
        embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Force CPU to save memory
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load FAISS database
        print("Loading FAISS index...")
        _db = FAISS.load_local(".", embeddings_model, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded successfully")
        
        # Initialize LLM with conservative settings
        modelName = "HuggingFaceH4/zephyr-7b-beta"
        print("Initializing LLM...")
        
        llm = HuggingFaceEndpoint(
            repo_id=modelName,
            temperature=0.2,
            max_new_tokens=100,  # Reduced for memory
            stop_sequences=["Question:", "\n\n\n", "Context:"],
            timeout=30  # Add timeout
        )
        
        # Define prompt
        prompt_template = """You are a helpful assistant that answers questions based ONLY on the provided context.
Context: {context}
Question: {question}
Instructions: Respond with exactly one word: "Yes" or "No".
- Use ONLY the information in the context above.
- If the context does not provide a clear answer, respond with "No".
- Do NOT provide any explanations, just the single word answer.

Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create RetrievalQA chain with reduced retrieval
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_db.as_retriever(search_kwargs={'k': 2}),  # Reduced from 3 to 2
            chain_type_kwargs={'prompt': PROMPT},
            return_source_documents=True
        )
        
        print("✅ QA Chain initialized successfully")
        
        # Force garbage collection
        gc.collect()
        
        return _qa_chain
        
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        traceback.print_exc()
        raise

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
    
    return query.strip()[:500]  # Limit query length

def get_answer(query: str):
    """Main function to get answers from the QA system"""
    try:
        if not query or not query.strip():
            return {"error": "Empty query provided"}
        
        cleaned_query = clean_query(query)
        print(f"Processing query: {cleaned_query[:100]}...")  # Truncate for logging
        
        # Get QA chain (lazy loaded)
        qa_chain = get_qa_chain()
        
        # Get response
        response = qa_chain.invoke({"query": cleaned_query})
        raw_answer = response.get("result", "").strip()
        
        # Process sources
        sources = []
        if "source_documents" in response and response["source_documents"]:
            for i, doc in enumerate(response["source_documents"][:2]):  # Limit sources
                sources.append({
                    "content": doc.page_content[:500],  # Truncate content
                    "metadata": doc.metadata,
                    "source_id": i + 1
                })
        
        print(f"Generated answer: {raw_answer}")
        print(f"Sources found: {len(sources)}")
        
        # Force garbage collection after processing
        gc.collect()
        
        return {
            "answer": raw_answer,
            "sources": sources,
            "query": cleaned_query
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # Force garbage collection on error
        gc.collect()
        
        return {"error": error_msg}

# Initialize system on import (for production)
def initialize_system():
    """Initialize the system - call this once on startup"""
    try:
        print("Initializing system...")
        authorized = initialize_drive_and_files()
        print(f"Drive authorized: {authorized}")
        
        # Pre-load the QA chain
        get_qa_chain()
        print("✅ System initialization complete")
        
    except Exception as e:
        print(f"❌ System initialization failed: {str(e)}")
        raise

# Health check function
def health_check():
    """Simple health check for the application"""
    try:
        return {
            "status": "healthy",
            "qa_chain_loaded": _qa_chain is not None,
            "db_loaded": _db is not None
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