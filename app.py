import streamlit as st
import pdfplumber
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to call the Groq API
def call_groq_api(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"  # Updated endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "llama-3.3-70b-versatile",  # Update as needed
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

# Functions to generate outputs
def generate_summary(text, api_key):
    prompt = f"Summarize the following lecture content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_faq(text, api_key):
    prompt = f"Generate 5 frequently asked questions with answers based on the following lecture content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_practice_questions(text, api_key):
    prompt = f"Generate 5 practice questions with answers based on the following lecture content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_quiz(text, api_key):
    prompt = f"Generate a multiple-choice quiz with 5 questions based on the following lecture content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

# Streamlit App
def main():
    st.title("Automated Lecture Summarizer & Quiz Generator")
    st.write("Upload your lecture material (PDF or text) to receive a summary, FAQs, practice questions, and a quiz.")

    uploaded_file = st.file_uploader("Upload Lecture Material", type=["pdf", "txt"])
    lecture_text = ""

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == "pdf":
            lecture_text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "txt":
            lecture_text = uploaded_file.read().decode("utf-8")

    if lecture_text:
        st.subheader("Extracted Lecture Text (first 500 characters)")
        st.write(lecture_text[:500] + "...")
        
        if API_KEY:
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(lecture_text, API_KEY)
                    st.subheader("Lecture Summary")
                    st.write(summary)
            
            if st.button("Generate FAQs"):
                with st.spinner("Generating FAQs..."):
                    faq = generate_faq(lecture_text, API_KEY)
                    st.subheader("FAQs")
                    st.write(faq)
            
            if st.button("Generate Practice Questions"):
                with st.spinner("Generating practice questions..."):
                    practice_questions = generate_practice_questions(lecture_text, API_KEY)
                    st.subheader("Practice Questions")
                    st.write(practice_questions)
            
            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(lecture_text, API_KEY)
                    st.subheader("Quiz")
                    st.write(quiz)
        else:
            st.warning("Please set your GROQ_API_KEY in your .env file.")
    else:
        st.info("Please upload a lecture file to get started.")

if __name__ == "__main__":
    main()
