from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Request
# from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import markdown
import pandas as pd
from pathlib import Path
import shutil
import os
import re
import json
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

# For Question W3-Q7

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


# For Question GA4-Q3

def get_wikipedia_url(country: str) -> str:
    """
    Given a country name, returns the Wikipedia URL for the country.
    """
    return f"https://en.wikipedia.org/wiki/{country}"

def extract_headings_from_html(html: str) -> list:
    """
    Extract all headings (H1 to H6) from the given HTML and return a list.
    """
    soup = BeautifulSoup(html, "html.parser")
    headings = []

    # Loop through all the heading tags (H1 to H6)
    for level in range(1, 7):
        for tag in soup.find_all(f'h{level}'):
            headings.append((level, tag.get_text(strip=True)))

    return headings

def generate_markdown_outline(headings: list) -> str:
    """
    Converts the extracted headings into a markdown-formatted outline.
    """
    markdown_outline = "## Contents\n\n"
    for level, heading in headings:
        markdown_outline += "#" * level + f" {heading}\n\n"
    return markdown_outline

@app.get("/api/outline")
async def get_country_outline(country: str):
    """
    API endpoint that returns the markdown outline of the given country Wikipedia page.
    """
    if not country:
        raise HTTPException(status_code=400, detail="Country parameter is required")

    # Fetch Wikipedia page
    url = get_wikipedia_url(country)
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Error fetching Wikipedia page: {e}")

    # Extract headings and generate markdown outline
    headings = extract_headings_from_html(response.text)
    if not headings:
        raise HTTPException(status_code=404, detail="No headings found in the Wikipedia page")

    markdown_outline = generate_markdown_outline(headings)
    return JSONResponse(content={"outline": markdown_outline})

# For Question GA3-Q8

def get_ticket_status(ticket_id: int):
    return {"ticket_id": ticket_id}

def schedule_meeting(date: str, time: str, meeting_room: str):
    return {"date": date, "time": time, "meeting_room": meeting_room}

def get_expense_balance(employee_id: int):
    return {"employee_id": employee_id}

def calculate_performance_bonus(employee_id: int, current_year: int):
    return {"employee_id": employee_id, "current_year": current_year}

def report_office_issue(issue_code: int, department: str):
    return {"issue_code": issue_code, "department": department}

@app.get("/execute")
async def execute_query(q: str):
    try:
        query = q.lower()
        pattern_debug_info = {}

        # Ticket status pattern
        if re.search(r"ticket.*?\d+", query):
            ticket_id = int(re.search(r"ticket.*?(\d+)", query).group(1))
            return {"name": "get_ticket_status", "arguments": json.dumps({"ticket_id": ticket_id})}
        pattern_debug_info["ticket_status"] = re.search(r"ticket.*?\d+", query) is not None

        # Meeting scheduling pattern
        if re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{2}:\d{2})", query)
            room_match = re.search(r"room\s*([A-Za-z0-9]+)", query, re.IGNORECASE)
            if date_match and time_match and room_match:
                return {"name": "schedule_meeting", "arguments": json.dumps({
                    "date": date_match.group(1),
                    "time": time_match.group(1),
                    "meeting_room": f"Room {room_match.group(1).capitalize()}"
                })}
        pattern_debug_info["meeting_scheduling"] = re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE) is not None

        # Expense balance pattern
        if re.search(r"expense", query, re.IGNORECASE):
            emp_match = re.search(r"(?:employee|emp)\s*(\d+)", query, re.IGNORECASE)  # Match 'employee' or 'emp'

            if emp_match:
                return {
                    "name": "get_expense_balance",
                    "arguments": json.dumps({
                        "employee_id": int(emp_match.group(1)),
                    })
                }

        pattern_debug_info["expense_balance"] = re.search(r"expense", query) is not None

        # Performance bonus pattern
        if re.search(r"bonus", query, re.IGNORECASE):
            emp_match = re.search(r"emp(?:loyee)?\s*(\d+)", query, re.IGNORECASE)
            year_match = re.search(r"\b(2024|2025)\b", query)
            if emp_match and year_match:
                return {"name": "calculate_performance_bonus", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1)),
                    "current_year": int(year_match.group(1))
                })}
        pattern_debug_info["performance_bonus"] = re.search(r"bonus", query, re.IGNORECASE) is not None

        # Office issue pattern
        if re.search(r"(office issue|report issue)", query, re.IGNORECASE):
            code_match = re.search(r"(issue|number|code)\s*(\d+)", query, re.IGNORECASE)
            dept_match = re.search(r"(in|for the)\s+(\w+)(\s+department)?", query, re.IGNORECASE)
            if code_match and dept_match:
                return {"name": "report_office_issue", "arguments": json.dumps({
                    "issue_code": int(code_match.group(2)),
                    "department": dept_match.group(2).capitalize()
                })}
        pattern_debug_info["office_issue"] = re.search(r"(office issue|report issue)", query, re.IGNORECASE) is not None

        raise HTTPException(status_code=400, detail=f"Could not parse query: {q}")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse query: {q}. Error: {str(e)}. Pattern matches: {pattern_debug_info}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)