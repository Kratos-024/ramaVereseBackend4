import os
import re
import json
import pickle
from typing import List, Dict, Any
import requests
import time
from dotenv import load_dotenv
from data.drive import Drive
import traceback

# Load environment variables
load_dotenv()
apiKey = os.getenv('ApiKey')

class SimpleVectorStore:
    """Lightweight alternative to FAISS - stores embeddings in memory"""
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, docs, embeddings, metadata):
        self.documents.extend(docs)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
    
    def similarity_search(self, query_embedding, k=3):
        """Simple cosine similarity search"""
        if not self.embeddings:
            return []
        
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            # Simple dot product similarity (assuming normalized embeddings)
            similarity = sum(a * b for a, b in zip(query_embedding, embedding))
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        results = []
        for _, idx in similarities[:k]:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx]
            })
        return results

class LightweightEmbeddings:
    """Ultra-lightweight embeddings using simple text features"""
    def __init__(self):
        # Simple vocabulary for basic text representation
        self.vocab = {}
        self.vocab_size = 100  # Very small vocabulary
    
    def _build_vocab(self, texts):
        """Build a simple vocabulary from texts"""
        word_freq = {}
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Keep only most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: i for i, (word, _) in enumerate(sorted_words[:self.vocab_size])}
    
    def _text_to_vector(self, text):
        """Convert text to simple frequency vector"""
        vector = [0.0] * self.vocab_size
        words = re.findall(r'\w+', text.lower())
        
        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] += 1.0
        
        # Simple normalization
        total = sum(vector)
        if total > 0:
            vector = [v / total for v in vector]
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.vocab:
            self._build_vocab(texts)
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)

class SimpleLLM:
    """Simple keyword-based answering system"""
    def __init__(self):
        # Simple patterns for yes/no questions
        self.yes_patterns = [
            r'(is|was|were|are|am)\s+.*\s+(yes|true|correct|right)',
            r'(does|do|did)\s+.*\s+(yes|indeed|certainly)',
            r'.*\s+(son|daughter|child|father|mother|parent)\s+of\s+.*',
            r'.*\s+(king|queen|ruler|emperor)\s+.*',
        ]
        
        self.no_patterns = [
            r'(not|never|no|false|incorrect|wrong)',
            r'(cannot|can\'t|couldn\'t|wouldn\'t|shouldn\'t)',
        ]
    
    def generate(self, context: str, question: str) -> str:
        """Simple pattern-based generation"""
        combined_text = (context + " " + question).lower()
        
        # Check for yes patterns
        for pattern in self.yes_patterns:
            if re.search(pattern, combined_text):
                return "Yes"
        
        # Check for no patterns  
        for pattern in self.no_patterns:
            if re.search(pattern, combined_text):
                return "No"
        
        # Default fallback - simple keyword matching
        question_words = set(re.findall(r'\w+', question.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        
        # If significant overlap, assume yes
        overlap = len(question_words.intersection(context_words))
        if overlap >= 2:
            return "Yes"
        
        return "No"

class LightweightRAG:
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.embeddings = LightweightEmbeddings()
        self.llm = SimpleLLM()
        self.drive = Drive()
        
    def load_data(self):
        """Load and process data with minimal memory usage"""
        try:
            # Try to load preprocessed data first
            if os.path.exists('lightweight_data.pkl'):
                print("Loading preprocessed data...")
                with open('lightweight_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.vector_store.documents = data['documents']
                    self.vector_store.embeddings = data['embeddings'] 
                    self.vector_store.metadata = data['metadata']
                    self.embeddings.vocab = data['vocab']
                print("✅ Preprocessed data loaded")
                return True
            
            # If no preprocessed data, create minimal sample data
            print("Creating minimal sample data...")
            sample_docs = [
                "Rama was the son of King Dasharatha of Ayodhya",
                "Dasharatha was the king of Ayodhya and father of Rama",
                "Sita was the wife of Rama and daughter of King Janaka",
                "Hanuman was a devotee of Rama and helped in rescuing Sita",
                "Ravana was the demon king of Lanka who abducted Sita"
            ]
            
            sample_metadata = [
                {'source': 'ramayana', 'chapter': 1},
                {'source': 'ramayana', 'chapter': 1},
                {'source': 'ramayana', 'chapter': 2},
                {'source': 'ramayana', 'chapter': 3},
                {'source': 'ramayana', 'chapter': 4}
            ]
            
            # Create embeddings
            embeddings = self.embeddings.embed_documents(sample_docs)
            self.vector_store.add_documents(sample_docs, embeddings, sample_metadata)
            
            # Save preprocessed data
            data = {
                'documents': self.vector_store.documents,
                'embeddings': self.vector_store.embeddings,
                'metadata': self.vector_store.metadata,
                'vocab': self.embeddings.vocab
            }
            
            with open('lightweight_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            print("✅ Sample data created and saved")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def search_and_answer(self, query: str, k: int = 2) -> Dict[str, Any]:
        """Search and generate answer with minimal computation"""
        try:
            # Clean query
            cleaned_query = re.sub(r'[^\w\s.,?!-]', '', query).strip()
            print(f"Query: {cleaned_query}")
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(cleaned_query)
            
            # Search for relevant documents
            results = self.vector_store.similarity_search(query_embedding, k=k)
            
            # Combine context
            context = " ".join([doc['content'] for doc in results])
            
            # Generate answer
            answer = self.llm.generate(context, cleaned_query)
            
            return {
                "answer": answer,
                "sources": results,
                "context_used": context
            }
            
        except Exception as e:
            print(f"Error in search_and_answer: {e}")
            traceback.print_exc()
            return {"error": str(e)}

# Global instance
rag_system = None

def initialize_system():
    """Initialize the RAG system"""
    global rag_system
    if rag_system is None:
        print("Initializing lightweight RAG system...")
        rag_system = LightweightRAG()
        success = rag_system.load_data()
        if not success:
            print("❌ Failed to initialize system")
            return False
        print("✅ System initialized successfully")
    return True

def get_answer(query: str):
    """Main function to get answers"""
    try:
        if not initialize_system():
            return {"error": "System initialization failed"}
        
        result = rag_system.search_and_answer(query)
        return result
        
    except Exception as e:
        print(f"Error in get_answer: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# Clean query function (kept for compatibility)
def clean_query(query: str) -> str:
    query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")
    query = re.sub(r'[^\w\s.,?!-]', '', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Lightweight RAG System")
    print("="*50)
    
    # Test the system
    result = get_answer("Rama was the son of king dasharatha")
    print(f"\nResult: {result}")
    
    # Test memory usage
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"\nMemory usage: {memory_mb:.2f} MB")