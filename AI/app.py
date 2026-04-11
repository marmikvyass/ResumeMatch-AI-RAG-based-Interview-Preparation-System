from fastapi import FastAPI, UploadFile, File, Form
import os

os.makedirs("temp", exist_ok=True)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Resume Analyzer API running go to /docs for API documentation"}

@app.post('/upload')
async def upload_resume(file:UploadFile = File(...)):
    from injest import injest_pdf
    import shutil
    from rag import reset_vector
    
    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    reset_vector()
    injest_pdf(path)
    return {"status": "resume indexed"}

@app.post('/analyze')
async def analyze_resume(job_desc  :str = Form(...)):
    from rag import generate_questions
    res = generate_questions(job_desc)
    return res

    

    
    