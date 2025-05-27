# !pip install langchain langchain_huggingface langchain_core langchain_community
# !pip install faiss-cpu
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import json
import faiss
import numpy as np
import pickle




#Loading ramayana dataset (JSON)
def load_json():
  with open(ramaDataPath,'r') as f:
    data = json.load(f)
  return data
data = load_json()


# Create documents
def create_document(data):
    return [
        Document(
            page_content=item['Translation'],
            metadata={"Shloka": item.get("Shloka"), "Sarga": item.get("Sarga"),"Kanda": item.get("Kanda"),"Chapter": item.get("Chapter")}
        )
        for item in data
    ]
documents = create_document(data)



    
# Get embedding model
embedding_model = "BAAI/bge-small-en"
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings

embedding_model = get_embedding_model()
print(embedding_model)


#Creating embeddings
batch_size = 256
all_embeddings = []

for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    texts = [doc.page_content for doc in batch_docs]

    batch_embeds = embedding_model.embed_documents(texts)
    all_embeddings.extend(batch_embeds)

    print(f"Processed {min(i + batch_size, len(documents))} / {len(documents)} documents")

print("All embeddings done! Building FAISS index...")

#Creating index.Faiss

embedding_matrix = np.array(all_embeddings).astype('float32')

#Building FAISS index (L2 distance)
embedding_dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)

#Saveing FAISS index
faiss.write_index(index, "/content/drive/MyDrive/dataset/ramayana_index.faiss")


with open("/content/drive/MyDrive/dataset/ramayana_documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("✅ FAISS index and documents saved!")


index = faiss.read_index("/content/drive/MyDrive/dataset/ramayana_index.faiss")

with open("/content/drive/MyDrive/dataset/ramayana_documents.pkl", "rb") as f:
    documents = pickle.load(f)
    

vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("/content/drive/MyDrive/dataset")
