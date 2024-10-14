import os
import time
import joblib
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
import easyocr
import numpy as np
import pdf2image
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
api_key = GOOGLE_API_KEY

# Create a data folder if it doesn't exist
os.makedirs('data/', exist_ok=True)

def get_file_text_from_local():
    text = ""
    reader = easyocr.Reader(['en'])
    pdf_files = list(Path('chatbot_context').rglob('*.pdf'))  # Recursively find all PDF files

    for file_path in pdf_files:
        file = Path(file_path)
        # st.write(f"Processing file: {file.name}")  # Debug output to show which file is being processed 
        start_time = time.time()
        text += f"\n--- The text below is from {file.name} ---\n"
        check = False
        try:
            # Reading the PDF file
            with file.open('rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text
                        check = True
        except Exception as e:
            st.write(f"Error reading text from {file.name}: {e}")  # Debug output for error handling

        # If text extraction from PDF fails, try OCR
        if not check:
            try:
                file_bytes = file.read_bytes()
                images = pdf2image.convert_from_bytes(file_bytes)
                for page in images:
                    results = reader.readtext(np.array(page))
                    for i in results:
                        text += i[1] + " "
            except Exception as e:
                st.write(f"Error processing OCR for {file.name}: {e}")  # Debug output for error handling
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        st.write(f"Processing {file.name} took {int(minutes)} minute(s) and {seconds:.2f} seconds.")
    return text


# VECTORISASI
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(r"faiss_index")

# Main application logic
st.title("🛠️ Admin Panel")

# Simple login form
username = st.text_input("Username")
password = st.text_input("Password", type="password")


if username == "pox" and password == "pox":
    if st.button("UPDATE CONTEXT/FAISS"):
        with st.status("Updating the FAISS...."):
            start_time = time.time()
            raw_text = get_file_text_from_local()
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, api_key)
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            st.write(f"Total Processing took {int(minutes)} minute(s) and {seconds:.2f} seconds.")
else:
    if username or password:
        st.error("Invalid username or password. Please try again.")
