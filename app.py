from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from rag_engine import process_pdf, retrieve, generate_answer, generate_summary
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

CURRENT_STORE = None
CURRENT_FILENAME = None

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",
{
    "request": request,
    "answer": None,
    "filename": CURRENT_FILENAME
})

@app.post("/upload_pdf/", response_class = HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    global CURRENT_STORE, CURRENT_FILENAME
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    CURRENT_STORE = process_pdf(file_path)
    CURRENT_FILENAME = file.filename

    return templates.TemplateResponse("index.html",
    {
        "request": request,
        "answer" : "PDF uploaded and processed successfully",
        "filename" : CURRENT_FILENAME
    })

@app.post("/ask/", response_class = HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    global CURRENT_STORE
    if CURRENT_STORE is None:
        return templates.TemplateResponse("index.html",{
            "request": request,
            "answer" : "No PDF uploaded.",
            "filename":None
        })
    retrieved_chunks = retrieve(
        question,
        CURRENT_STORE["index"],
        CURRENT_STORE["chunks"]
    )
    
    retrieved_chunks = retrieved_chunks[:4]

    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\nChunk {i+1}:\n", chunk[:300])
    context = "\n\n".join(retrieved_chunks)

    print("Context length:", len(context))

    if "summary" in question.lower():
        full_context = " ".join(CURRENT_STORE["chunks"])
        answer = generate_summary(full_context)
    else:
        answer = generate_answer(question, context)
    
    return templates.TemplateResponse("index.html",{
        "request": request,
        "answer": answer,
        "filename": CURRENT_FILENAME
    })