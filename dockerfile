FROM python:3.9-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils

RUN pip install fastapi uvicorn pymupdf pytesseract pdf2image

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]