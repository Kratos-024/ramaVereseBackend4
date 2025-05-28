from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
from drive import Drive
load_dotenv()
apiKey = os.getenv('ApiKey')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = apiKey
drive = Drive()
authorized = drive.authorize()

faissFileid = "1CcT7gZMQrkZB6S4WalAFEb_rIZMJOHXx"
pklFileid = "1WJpJIvpERopl9ANDuq-a-Vi6PfHrS5SB"

faiss_filename = "index.faiss"
pkl_filename = "index.pkl"

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

embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model)

# Load vector database
db = FAISS.load_local(".", embeddings_model, allow_dangerous_deserialization=True)
modelName = "HuggingFaceH4/zephyr-7b-beta"
# Initialize LLM with better parameters
llm = HuggingFaceEndpoint(
    repo_id=modelName,
    temperature=0.2,
    max_new_tokens=150,
    stop_sequences=["Question:", "\n\n\n", "Context:"]
)

# Improved prompt template
prompt_template = """You are a helpful assistant that answers questions based ONLY on the provided context.

Context: {context}

Question: {question}

Instructions: Respond with exactly one word: "Yes" or "No".
- Use ONLY the information in the context above.
- If the context does not provide a clear answer, respond with "No".
- Do NOT provide any explanations, just the single word answer.

Answer:"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={'prompt': PROMPT},
    return_source_documents=True
)

def clean_query(query: str) -> str:
    """Minimal cleaning to preserve query meaning"""
    # Remove only dangerous characters, keep case and punctuation
    query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")
    query = re.sub(r'[^\w\s.,?!-]', '', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()
import traceback

def get_answer(query: str):
    try:
        cleaned_query = clean_query(query)
        print(f"Original Question: {query}")
        print(f"Cleaned Question: {cleaned_query}")
        
        # Get response from QA chain
        response = qa_chain.invoke({"query": cleaned_query})
        raw_answer = response["result"].strip()
        
        print(f"Raw LLM Response: {raw_answer}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

get_answer("Hanuman was the son of Ravana.")

