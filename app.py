from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Request
# from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import shutil
import os
import requests
from typing import Dict, List
import numpy as np
import traceback

app = FastAPI()

load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set. Make sure it's defined in your .env file.")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}



@app.post("/name")
def get_students(class_: list[str] = Query(None, alias="class"), file:UploadFile = File(...)):
    temp_dir = Path("/tmp")  # Render allows using /tmp
    temp_file_path = temp_dir / file.filename  # Full path for saving

    # Save the uploaded file to `/tmp/`
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(temp_file_path)

    if class_:
        filtered_df = df[df["class"].isin(class_)]
    else:
        filtered_df = df

    # Convert to dictionary list
    students = filtered_df.to_dict(orient="records")
    return {"students": students}

@app.post("/api")
def get_students(class_: list[str] = Query(None, alias="class")):
    # temp_dir = Path("/tmp")  # Render allows using /tmp
    # temp_file_path = temp_dir / file.filename  # Full path for saving
    temp_file_path = "/home/abhijit/Sample/data/q-fastapi (1).csv"

    # Save the uploaded file to `/tmp/`
    # with open(temp_file_path, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(temp_file_path)

    if class_:
        filtered_df = df[df["class"].isin(class_)]
    else:
        filtered_df = df

    # Convert to dictionary list
    students = filtered_df.to_dict(orient="records")
    return {"students": students}

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else np.dot(a, b) / (norm_a * norm_b)

@app.post("/similarity")
async def get_similar_docs(request: Request, request_body: Dict):
    try:
        docs: List[str] = request_body.get("docs")
        query: str = request_body.get("query")

        if not docs or not query:
            raise HTTPException(status_code=400, detail="Missing 'docs' or 'query' in request body")

        input_texts = [query] + docs  # Combine query and docs for embeddings request

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        data = {"model": "text-embedding-3-small", "input": input_texts}
        embeddings_response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers=headers,
            json=data
        )

        embeddings_response.raise_for_status()
        embeddings_data = embeddings_response.json()

        query_embedding = embeddings_data['data'][0]['embedding']
        doc_embeddings = [emb['embedding'] for emb in embeddings_data['data'][1:]]

        similarities = [(i, cosine_similarity(query_embedding, doc_embeddings[i]), docs[i]) for i in range(len(docs))]
        ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_matches = [doc for _, _, doc in ranked_docs[:min(3, len(ranked_docs))]]

        return {"matches": top_matches}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with AI Proxy: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)