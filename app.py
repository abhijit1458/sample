from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from pathlib import Path
import shutil

app = FastAPI()

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




if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)