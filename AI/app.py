from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from rag import load_pdf,generate_questions

os.makedirs("temp", exist_ok=True)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Resume Analyzer API running go to /docs for API documentation"}

@app.post('/upload')
async def upload_resume(file:UploadFile = File(...)):

    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    load_pdf(path)
    return {"status": "resume indexed"}

@app.post('/analyze')
async def analyze_resume(job_desc  :str = Form(...)):
    res = generate_questions(job_desc)
    return res

    

    
    