from fastapi import FastAPI, UploadFile, File, Form
from rag import generate_questions
import shutil
import os

os.makedirs("temp", exist_ok=True)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Resume Analyzer API running go to /docs for API documentation"}

@app.post('/analyze')
async def analyze_resume(
    file:UploadFile = File(...),
    job_desc : str = Form(...)
):
    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    res = generate_questions(path, job_desc)
    return res