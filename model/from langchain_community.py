from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint  # Updated import
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
from data.drive import Drive
import traceback
import requests
import time

# Load environment variables
load_dotenv()
apiKey = os.getenv('ApiKey')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apiKey

# Authorize Google Drive
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
else:
    print(f"{faiss_filename} already exists, skipping download.")

if not os.path.isfile(pkl_filename):
    drive.download_file(authorized, pklFileid, pkl_filename)
else:
    print(f"{pkl_filename} already exists, skipping download.")

print("Authorized", authorized)

embedding_model = "BAAI/bge-small-en"

# Custom embedding class with better error handling
class RobustHuggingFaceInferenceAPIEmbeddings:
    def __init__(self, model_name, api_key, max_retries=3, retry_delay=1):
        self.model_name = model_name
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def _make_request(self, payload):
        """Make API request with retry logic and better error handling"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                
                # Check if response is successful
                if response.status_code == 200:
                    try:
                        return response.json()
                    except requests.exceptions.JSONDecodeError:
                        print(f"Invalid JSON response: {response.text}")
                        if attempt < self.max_retries - 1:
                            print(f"Retrying in {self.retry_delay} seconds...")
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            raise Exception(f"Failed to get valid JSON after {self.max_retries} attempts")
                
                elif response.status_code == 503:
                    print(f"Model loading, attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        raise Exception("Model failed to load after multiple attempts")
                
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    raise Exception(f"API request failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
        
        raise Exception(f"Failed after {self.max_retries} attempts")
    
    def embed_documents(self, texts):
        """Embed multiple documents"""
        try:
            payload = {"inputs": texts}
            embeddings = self._make_request(payload)
            
            # Handle different response formats
            if isinstance(embeddings, list) and len(embeddings) > 0:
                if isinstance(embeddings[0], list):
                    return embeddings
                elif isinstance(embeddings[0], dict) and 'embedding' in embeddings[0]:
                    return [item['embedding'] for item in embeddings]
            
            raise Exception(f"Unexpected embedding format: {type(embeddings)}")
            
        except Exception as e:
            print(f"Embedding error: {e}")
            raise
    
    def embed_query(self, text):
        """Embed a single query"""
        return self.embed_documents([text])[0]

# Alternative: Use local embeddings as fallback
def get_local_embeddings():
    """Fallback to local embeddings if API fails"""
    try:
        from sentence_transformers import SentenceTransformer
        
        class LocalEmbeddings:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        return LocalEmbeddings()
    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None

# Try to create embeddings model with fallback
try:
    print("Trying Hugging Face API embeddings...")
    embeddings_model = RobustHuggingFaceInferenceAPIEmbeddings(
        model_name=embedding_model,
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_retries=3,
        retry_delay=2
    )
    
    # Test the embeddings
    test_embedding = embeddings_model.embed_query("test")
    print(f"✅ Hugging Face API embeddings working! Embedding dimension: {len(test_embedding)}")
    
except Exception as e:
    print(f"❌ Hugging Face API embeddings failed: {e}")
    print("Trying local embeddings as fallback...")
    
    embeddings_model = get_local_embeddings()
    if embeddings_model is None:
        print("❌ No embedding model available. Please check your API key or install sentence-transformers")
        exit(1)
    else:
        print("✅ Using local embeddings")

# Load FAISS index
try:
    db = FAISS.load_local(".", embeddings_model, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded successfully")
except Exception as e:
    print(f"❌ Failed to load FAISS index: {e}")
    exit(1)

# Define LLM with updated import
modelName = "HuggingFaceH4/zephyr-7b-beta"

llm = HuggingFaceEndpoint(
    repo_id=modelName,
    temperature=0.2,
    max_new_tokens=150,
    stop_sequences=["Question:", "\n\n\n", "Context:"]
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

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={'prompt': PROMPT},
    return_source_documents=True
)

# Clean query
def clean_query(query: str) -> str:
    query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")
    query = re.sub(r'[^\w\s.,?!-]', '', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

# Main answer function
def get_answer(query: str):
    try:
        cleaned_query = clean_query(query)
        print(f"Original Question: {query}")
        print(f"Cleaned Question: {cleaned_query}")

        response = qa_chain.invoke({"query": cleaned_query})
        raw_answer = response["result"].strip()

        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        print(f"Raw LLM Response: {raw_answer}")
        print(f"Sources found: {len(sources)}")

        return {
            "answer": raw_answer,
            "sources": sources
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing RAG System")
    print("="*50)
    result = get_answer("Rama was the son of king dasharatha")
    print(f"\nResult: {result}")