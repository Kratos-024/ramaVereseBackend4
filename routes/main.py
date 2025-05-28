from fastapi import FastAPI
from pydantic import BaseModel
from model.main import get_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class VerseRequest(BaseModel):
    verse: str
    
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)
@app.get("/")
def read_root():
    return {"message": "Hello world"}

@app.post("/get-verse")
def get_verse(request: VerseRequest):
    try:
        response = get_answer(request.verse) 
        return {"Answer": response["answer"],
                "Documents":response["sources"]}
    except Exception as e:
        return {"error": str(e)}
    
    
