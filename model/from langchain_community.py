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
from langchain.embeddings import HuggingFaceEmbeddings

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




print("Trying Hugging Face API embeddings...")
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model,
      
    )
    



db = FAISS.load_local(".", embeddings_model, allow_dangerous_deserialization=True)
print("✅ FAISS index loaded successfully")

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
