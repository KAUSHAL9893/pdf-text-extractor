from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import tempfile

app = FastAPI()

# Allow CORS (for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_methods=["POST"],
)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        # Try direct text extraction (digital PDFs)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        
        # Fallback to OCR if no text found (scanned PDFs)
        if not text.strip():
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_text(text)

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        # Process PDF
        text = extract_text_from_pdf(tmp_path)
        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Text chunking failed")
        
        # Generate embeddings
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Save FAISS index (optional)
        output_dir = "vectorstores"
        os.makedirs(output_dir, exist_ok=True)
        vectorstore.save_local(output_dir)
        
        # Return first 500 chars for demo
        return {
            "status": "success",
            "text": text[:500] + "...",
            "num_chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            os.unlink(tmp_path)  # Clean up

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)