import os
import json
import sqlite3
import numpy as np
import faiss
import requests
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Disable JIT profiling executor to avoid event-loop issues
torch._C._jit_set_profiling_executor(False)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

#############################################
# Database & Local Storage Setup (SQLite)
#############################################
DB_PATH = "metadata.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    filename TEXT,
                    extracted_text TEXT,
                    summary TEXT,
                    chunks TEXT
                )''')
    # Ensure "chunks" column exists (for legacy databases)
    c.execute("PRAGMA table_info(files)")
    columns = [col[1] for col in c.fetchall()]
    if "chunks" not in columns:
        c.execute("ALTER TABLE files ADD COLUMN chunks TEXT")
    conn.commit()
    conn.close()

init_db()

def save_file_metadata(subject, filename, extracted_text, summary, chunks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    chunks_str = json.dumps(chunks) if chunks else json.dumps([])
    c.execute('''INSERT INTO files (subject, filename, extracted_text, summary, chunks)
                 VALUES (?, ?, ?, ?, ?)''', (subject, filename, extracted_text, summary, chunks_str))
    conn.commit()
    conn.close()

def load_files_by_subject(subject):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename, extracted_text, summary, chunks FROM files WHERE subject=?", (subject,))
    rows = c.fetchall()
    conn.close()
    files = {}
    for row in rows:
        filename, extracted_text, summary, chunks_str = row
        files[filename] = {
            "extracted_text": extracted_text,
            "summary": summary,
            "chunks": json.loads(chunks_str)
        }
    return files

#############################################
# FAISS Global Index Setup (for all chunks)
#############################################
embedding_dim = 384  # for "all-MiniLM-L6-v2"
INDEX_PATH = "faiss.index"

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    if os.path.exists("faiss_mapping.json"):
        with open("faiss_mapping.json", "r", encoding="utf-8") as f:
            faiss_mapping = json.load(f)
    else:
        faiss_mapping = []
else:
    index = faiss.IndexFlatL2(embedding_dim)
    faiss_mapping = []

def update_faiss_index(embedding, meta):
    vec = np.array([embedding]).astype('float32')
    index.add(vec)
    faiss_mapping.append(meta)
    faiss.write_index(index, INDEX_PATH)
    with open("faiss_mapping.json", "w", encoding="utf-8") as f:
        json.dump(faiss_mapping, f, indent=2)

#############################################
# Embedding Model & Caching
#############################################
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

@st.cache_data
def compute_embedding(text):
    return model.encode([text])[0]

#############################################
# Document Chunking for Hierarchical Summarization
#############################################
def split_text(text, chunk_size=3000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def hierarchical_summary(text, api_key, chunk_size=3000, overlap=200):
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    chunk_summaries = []
    for chunk in chunks:
        chunk_sum = generate_summary(chunk, api_key)
        chunk_summaries.append(chunk_sum)
    combined_summary = "\n".join(chunk_summaries)
    final_summary = generate_summary(combined_summary, api_key)
    return final_summary

#############################################
# Folder and File Management (Local)
#############################################
SUBJECTS_DIR = "subjects"

def ensure_subjects_dir():
    if not os.path.exists(SUBJECTS_DIR):
        os.makedirs(SUBJECTS_DIR)

def get_subjects():
    ensure_subjects_dir()
    return [d for d in os.listdir(SUBJECTS_DIR) if os.path.isdir(os.path.join(SUBJECTS_DIR, d))]

def create_subject(subject_name):
    ensure_subjects_dir()
    subject_path = os.path.join(SUBJECTS_DIR, subject_name)
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    return subject_path

def save_file_to_subject(subject, uploaded_file):
    subject_path = os.path.join(SUBJECTS_DIR, subject)
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    file_path = os.path.join(subject_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

#############################################
# File Text Extraction Functions
#############################################
def extract_text_from_pdf(file_input):
    text = ""
    with pdfplumber.open(file_input) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_txt(file_input):
    return file_input.read().decode("utf-8")

def extract_text(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

#############################################
# SERP API Web Search Functions
#############################################
def perform_serp_search(query):
    params = {
        "q": query,
        "engine": "google",
        "api_key": SERP_API_KEY,
        "hl": "en",
        "gl": "us"
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

def search_study_resources(query):
    results = perform_serp_search(query)
    items = []
    if "organic_results" in results:
        for item in results["organic_results"]:
            title = item.get("title", "No Title")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            items.append({"title": title, "link": link, "snippet": snippet})
    return items

# New function: Generate a search query from context using Groq API
def generate_search_query(context, api_key):
    prompt = f"Based on the following educational context:\n{context[:3000]}\nGenerate a concise search query to find the most relevant study resources (books, videos, courses) for this content."
    return call_groq_api(prompt, api_key)

#############################################
# API Call Functions for Groq (with max_tokens)
#############################################
def call_groq_api(prompt, api_key, max_tokens=2000):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

def generate_summary(text, api_key):
    prompt = f"You are an intelligent educational assistant designed to help students understand complex content by summarizing it into clear, concise, and engaging explanations. Your task is to read the provided content and create a summary that highlights the most important concepts, definitions, examples, and takeaways. The summary should be structured in an easily digestible format—use bullet points or numbered lists when it enhances clarity. Aim for a tone that is formal yet conversational, and feel free to include clever humor where it aids comprehension without compromising professionalism. Ensure that the summary is tailored for a student audience, making the material accessible and reinforcing key learning objectives. If any part of the content is ambiguous, offer a brief explanation or ask clarifying questions to ensure accuracy:\n{text[:3000]}"
    return call_groq_api(prompt, api_key)

def generate_faq(text, api_key):
    prompt = f"You are an educational assistant with a knack for breaking down complex material into digestible pieces. Your task is to review the provided content and generate a list of 5 frequently asked questions (FAQs) along with detailed, yet succinct answers. Ensure that each FAQ is directly relevant to the content, addresses common points of confusion, and reinforces key concepts. Present your answers in a clear, structured format, using bullet points or numbered lists as needed. Feel free to add a touch of clever humor to keep the tone light, but maintain a formal and professional style that would help a student truly grasp the subject matter:\n{text[:3000]}"
    return call_groq_api(prompt, api_key)

def generate_practice_questions(text, api_key):
    prompt = f"You are an expert educational guide tasked with reinforcing learning through practice. Using the provided content, generate 5 well-crafted practice questions along with their corresponding answers. The questions should cover a broad range of the key concepts and details from the content, challenging the student in a way that promotes deeper understanding. Structure your output in an organized manner—each question followed immediately by its answer, using lists or clear separations. Aim to be both informative and engaging, using a touch of clever humor where appropriate, while ensuring that your tone remains formal and professional:\n{text[:3000]}"
    return call_groq_api(prompt, api_key)

def generate_structured_quiz(text, api_key):
    prompt = (
        "Generate a multiple-choice quiz with 5 questions based on the following content. "
        "Return the quiz in JSON format as an array of objects. Each object should have: "
        '"question": <string>, "options": [<option A>, <option B>, <option C>, <option D>], '
        '"answer": <the letter of the correct option (A, B, C, or D) or the full text>, '
        'and "explanation": <a brief explanation of why the answer is correct>.\n'
        f"{text[:3000]}"
    )
    response_text = call_groq_api(prompt, API_KEY)
    json_start = response_text.find('[')
    json_end = response_text.rfind(']')
    if json_start != -1 and json_end != -1:
        json_text = response_text[json_start:json_end+1]
    else:
        json_text = response_text.strip()
    try:
        quiz_data = json.loads(json_text)
    except json.JSONDecodeError:
        st.error("Failed to parse quiz JSON. Response was:")
        st.error(response_text)
        quiz_data = []
    return quiz_data

def normalize_answer(q):
    ans = q.get("answer", "").strip().upper()
    options = q.get("options", [])
    if len(ans) == 1 and ans in ["A", "B", "C", "D"] and len(options) >= 4:
        mapping = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
        return mapping.get(ans, ans)
    return ans

def generate_chat_response(context_text, user_question, api_key):
    prompt = f"Based on the following context:\n{context_text[:3000]}\nAnswer the question below clearly and concisely:\n{user_question}"
    return call_groq_api(prompt, api_key)

#############################################
# Main Application
#############################################
def main():
    st.title("EduScribe RAG: Lecture Summarizer, QnA, and Resource Finder")
    
    # Sidebar: Subject Management
    st.sidebar.header("Subject Management")
    subjects = get_subjects()
    selected_subject = st.sidebar.selectbox("Select Subject", ["--None--"] + subjects)
    new_subject = st.sidebar.text_input("Create New Subject")
    if st.sidebar.button("Create Subject"):
        if new_subject.strip():
            create_subject(new_subject.strip())
            st.sidebar.success(f"Created subject: {new_subject.strip()}")
            st.experimental_rerun()
        else:
            st.sidebar.warning("Enter a valid subject name.")
    
    # Sidebar: Upload File to Subject (with ingestion/chunking)
    st.sidebar.subheader("Upload File to Subject")
    if selected_subject != "--None--":
        uploaded_subject_file = st.sidebar.file_uploader("Upload file for subject", key="subject_upload")
        if uploaded_subject_file:
            extracted_text = extract_text(uploaded_subject_file)
            # Use hierarchical summarization if text is long
            if len(extracted_text) > 5000:
                summary = hierarchical_summary(extracted_text, API_KEY)
            else:
                summary = generate_summary(extracted_text, API_KEY)
            # Ingest document to get chunks for retrieval
            chunks = []
            if len(extracted_text) > 5000:
                chunk_texts = split_text(extracted_text, chunk_size=2000, overlap=300)
                for ct in chunk_texts:
                    emb = compute_embedding(ct)
                    chunks.append({"text": ct, "embedding": emb.tolist()})
            else:
                emb = compute_embedding(extracted_text)
                chunks.append({"text": extracted_text, "embedding": emb.tolist()})
            subject_folder = os.path.join(SUBJECTS_DIR, selected_subject)
            file_path = os.path.join(subject_folder, uploaded_subject_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_subject_file.getbuffer())
            save_file_metadata(selected_subject, uploaded_subject_file.name, extracted_text, summary, chunks)
            # Update global FAISS index with each chunk embedding
            for idx, chunk in enumerate(chunks):
                update_faiss_index(np.array(chunk["embedding"], dtype="float32"), {"subject": selected_subject, "filename": uploaded_subject_file.name, "chunk_index": idx})
            st.sidebar.success(f"Saved {uploaded_subject_file.name} to {selected_subject}")
    else:
        st.sidebar.info("Select or create a subject to upload files.")
    
    # Main Tab Selection
    tab = st.radio("Choose Operation", ["Folder Operations", "File Operations", "Chat", "Web Search"])
    
    if tab == "Folder Operations":
        if selected_subject == "--None--":
            st.info("Select a subject folder for folder operations.")
        else:
            st.header(f"Subject: {selected_subject}")
            files_metadata = load_files_by_subject(selected_subject)
            aggregated_text = "\n".join([data.get("summary", "") for data in files_metadata.values() if data.get("summary")])
            st.subheader("Folder Aggregated Outputs")
            
            if st.button("Generate Aggregated Summary for Folder"):
                agg_summary = generate_summary(aggregated_text, API_KEY)
                st.session_state["folder_summary"] = agg_summary
            if "folder_summary" in st.session_state:
                st.markdown("**Folder Summary:**")
                st.write(st.session_state["folder_summary"])
            
            if st.button("Generate FAQs for Folder"):
                agg_faq = generate_faq(aggregated_text, API_KEY)
                st.session_state["folder_faq"] = agg_faq
            if "folder_faq" in st.session_state:
                st.markdown("**Folder FAQs:**")
                st.write(st.session_state["folder_faq"])
            
            if st.button("Generate Practice Questions for Folder"):
                agg_practice = generate_practice_questions(aggregated_text, API_KEY)
                st.session_state["folder_practice"] = agg_practice
            if "folder_practice" in st.session_state:
                st.markdown("**Folder Practice Questions:**")
                st.write(st.session_state["folder_practice"])
            
            if st.button("Generate Interactive Quiz for Folder"):
                st.session_state["folder_quiz"] = generate_structured_quiz(aggregated_text, API_KEY)
            if "folder_quiz" in st.session_state:
                folder_quiz = st.session_state["folder_quiz"]
                st.subheader("Interactive Quiz for Folder")
                for idx, q in enumerate(folder_quiz):
                    st.write(f"**Q{idx+1}: {q.get('question', 'No question provided')}**")
                    options = q.get("options", [])
                    if options and len(options) >= 4:
                        st.radio("Choose an option:", ["A", "B", "C", "D"], key=f"folder_q_{idx}")
                        st.markdown(f"**A.** {options[0]}")
                        st.markdown(f"**B.** {options[1]}")
                        st.markdown(f"**C.** {options[2]}")
                        st.markdown(f"**D.** {options[3]}")
                    else:
                        st.error("Incomplete quiz data.")
                if st.button("Submit Folder Quiz"):
                    score = 0
                    for idx, q in enumerate(folder_quiz):
                        correct_full = normalize_answer(q)
                        options = q.get("options", [])
                        user_letter = st.session_state.get(f"folder_q_{idx}")
                        if user_letter and options and len(options) >= 4:
                            mapping = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
                            user_choice = mapping.get(user_letter)
                        else:
                            user_choice = None
                        if user_choice == correct_full:
                            score += 1
                    st.success(f"Folder Quiz Score: {score} out of {len(folder_quiz)}")
                    for idx, q in enumerate(folder_quiz):
                        st.markdown(f"**Q{idx+1} Explanation:** {q.get('explanation', 'No explanation provided')}")
            
            st.subheader("Files in Subject")
            file_list = list(files_metadata.keys())
            selected_file = st.selectbox("Select a file", ["--None--"] + file_list)
            if selected_file != "--None--":
                file_data = files_metadata.get(selected_file, {})
                st.write(f"**File: {selected_file}**")
                st.write(file_data.get("extracted_text", "")[:500] + "...")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Summary for File"):
                        file_summary = generate_summary(file_data.get("extracted_text", ""), API_KEY)
                        st.session_state[f"{selected_file}_summary"] = file_summary
                    if st.session_state.get(f"{selected_file}_summary"):
                        st.markdown("**File Summary:**")
                        st.write(st.session_state.get(f"{selected_file}_summary"))
                with col2:
                    if st.button("Generate FAQs for File"):
                        file_faq = generate_faq(file_data.get("extracted_text", ""), API_KEY)
                        st.session_state[f"{selected_file}_faq"] = file_faq
                    if st.session_state.get(f"{selected_file}_faq"):
                        st.markdown("**File FAQs:**")
                        st.write(st.session_state.get(f"{selected_file}_faq"))
                if st.button("Generate Practice Questions for File"):
                    file_practice = generate_practice_questions(file_data.get("extracted_text", ""), API_KEY)
                    st.session_state[f"{selected_file}_practice"] = file_practice
                    st.markdown("**File Practice Questions:**")
                    st.write(file_practice)
                if st.button("Generate Interactive Quiz for File"):
                    st.session_state[f"{selected_file}_quiz"] = generate_structured_quiz(file_data.get("extracted_text", ""), API_KEY)
                if st.session_state.get(f"{selected_file}_quiz"):
                    file_quiz = st.session_state.get(f"{selected_file}_quiz")
                    st.subheader("Interactive Quiz for File")
                    for idx, q in enumerate(file_quiz):
                        st.write(f"**Q{idx+1}: {q.get('question', 'No question provided')}**")
                        options = q.get("options", [])
                        if options and len(options) >= 4:
                            st.radio("Choose an option:", ["A", "B", "C", "D"], key=f"file_q_{idx}")
                            st.markdown(f"**A.** {options[0]}")
                            st.markdown(f"**B.** {options[1]}")
                            st.markdown(f"**C.** {options[2]}")
                            st.markdown(f"**D.** {options[3]}")
                        else:
                            st.error("Incomplete quiz data.")
                    if st.button("Submit File Quiz"):
                        score = 0
                        for idx, q in enumerate(file_quiz):
                            correct_full = normalize_answer(q)
                            options = q.get("options", [])
                            user_letter = st.session_state.get(f"file_q_{idx}")
                            if user_letter and options and len(options) >= 4:
                                mapping = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
                                user_choice = mapping.get(user_letter)
                            else:
                                user_choice = None
                            if user_choice == correct_full:
                                score += 1
                        st.success(f"File Quiz Score: {score} out of {len(file_quiz)}")
                        for idx, q in enumerate(file_quiz):
                            st.markdown(f"**Q{idx+1} Explanation:** {q.get('explanation', 'No explanation provided')}")
    
    elif tab == "File Operations":
        st.header("Direct File Upload and Processing")
        uploaded_file_main = st.file_uploader("Upload Lecture Material (PDF or TXT)", key="main_file")
        if uploaded_file_main:
            lecture_text = extract_text(uploaded_file_main)
            st.write("**Extracted Text (first 500 characters):**")
            st.write(lecture_text[:500] + "...")
            if st.button("Generate Summary for Uploaded File"):
                up_sum = generate_summary(lecture_text, API_KEY)
                st.write("**Summary:**", up_sum)
            if st.button("Generate FAQs for Uploaded File"):
                up_faq = generate_faq(lecture_text, API_KEY)
                st.write("**FAQs:**", up_faq)
            if st.button("Generate Practice Questions for Uploaded File"):
                up_practice = generate_practice_questions(lecture_text, API_KEY)
                st.write("**Practice Questions:**", up_practice)
            if st.button("Generate Interactive Quiz for Uploaded File"):
                st.session_state["upload_quiz"] = generate_structured_quiz(lecture_text, API_KEY)
            if st.session_state.get("upload_quiz"):
                up_quiz = st.session_state.get("upload_quiz")
                st.subheader("Interactive Quiz for Uploaded File")
                for idx, q in enumerate(up_quiz):
                    st.write(f"**Q{idx+1}: {q.get('question', 'No question provided')}**")
                    options = q.get("options", [])
                    if options and len(options) >= 4:
                        st.radio("Choose an option:", ["A", "B", "C", "D"], key=f"upload_q_{idx}")
                        st.markdown(f"**A.** {options[0]}")
                        st.markdown(f"**B.** {options[1]}")
                        st.markdown(f"**C.** {options[2]}")
                        st.markdown(f"**D.** {options[3]}")
                    else:
                        st.error("Incomplete quiz data.")
                if st.button("Submit Uploaded File Quiz"):
                    score = 0
                    for idx, q in enumerate(up_quiz):
                        correct_full = normalize_answer(q)
                        options = q.get("options", [])
                        user_letter = st.session_state.get(f"upload_q_{idx}")
                        if user_letter and options and len(options) >= 4:
                            mapping = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
                            user_choice = mapping.get(user_letter)
                        else:
                            user_choice = None
                        if user_choice == correct_full:
                            score += 1
                    st.success(f"Uploaded File Quiz Score: {score} out of {len(up_quiz)}")
                    for idx, q in enumerate(up_quiz):
                        st.markdown(f"**Q{idx+1} Explanation:** {q.get('explanation', 'No explanation provided')}")
    
    elif tab == "Chat":
        st.header("Chat with Document Assistant")
        chat_context = st.selectbox("Chat Context", ["Folder", "Uploaded File"])
        if chat_context == "Folder":
            if selected_subject == "--None--":
                st.info("Select a subject folder for chat.")
            else:
                files_metadata = load_files_by_subject(selected_subject)
                aggregated_context = "\n".join([data.get("summary", "") for data in files_metadata.values() if data.get("summary")])
                if not aggregated_context:
                    aggregated_context = "\n".join([data.get("extracted_text", "")[:1000] for data in files_metadata.values()])
                st.write("Aggregated context loaded.")
                user_question = st.text_input("Enter your question about the folder content:", key="folder_chat")
                if st.button("Ask Folder Question"):
                    answer = generate_chat_response(aggregated_context, user_question, API_KEY)
                    st.markdown("**Answer:**")
                    st.write(answer)
        elif chat_context == "Uploaded File":
            uploaded_file_chat = st.file_uploader("Upload a file for chat", key="chat_file")
            if uploaded_file_chat:
                file_ext = uploaded_file_chat.name.split('.')[-1].lower()
                if file_ext == "pdf":
                    file_text = extract_text_from_pdf(uploaded_file_chat)
                elif file_ext == "txt":
                    file_text = extract_text_from_txt(uploaded_file_chat)
                else:
                    file_text = ""
                st.write("File context loaded.")
                user_question = st.text_input("Enter your question about the file content:", key="file_chat")
                if st.button("Ask File Question"):
                    answer = generate_chat_response(file_text, user_question, API_KEY)
                    st.markdown("**Answer:**")
                    st.write(answer)
    
    elif tab == "Web Search":
        st.header("Study Resource Web Search")
        # Provide a choice for context source
        context_source = st.selectbox("Select Context Source", ["Manual Query", "Use Subject Context", "Use Uploaded File Context"])
        if context_source == "Manual Query":
            search_query = st.text_input("Enter your search query for study resources:", key="web_search")
        elif context_source == "Use Subject Context":
            if selected_subject == "--None--":
                st.info("Select a subject for context-based web search.")
                search_query = ""
            else:
                files_metadata = load_files_by_subject(selected_subject)
                aggregated_context = "\n".join([data.get("summary", "") for data in files_metadata.values() if data.get("summary")])
                if not aggregated_context:
                    aggregated_context = "\n".join([data.get("extracted_text", "")[:1000] for data in files_metadata.values()])
                st.write("Using aggregated subject context.")
                search_query = generate_search_query(aggregated_context, API_KEY)
                st.write("Generated Search Query:", search_query)
        elif context_source == "Use Uploaded File Context":
            uploaded_file_search = st.file_uploader("Upload a file for search context", key="search_file")
            if uploaded_file_search:
                file_ext = uploaded_file_search.name.split('.')[-1].lower()
                if file_ext == "pdf":
                    file_text = extract_text_from_pdf(uploaded_file_search)
                elif file_ext == "txt":
                    file_text = extract_text_from_txt(uploaded_file_search)
                else:
                    file_text = ""
                st.write("Using uploaded file context.")
                search_query = generate_search_query(file_text, API_KEY)
                st.write("Generated Search Query:", search_query)
            else:
                search_query = ""
        if search_query:
            if st.button("Search Web"):
                results = search_study_resources(search_query)
                if "error" in results:
                    st.error("Web search error: " + results["error"])
                else:
                    st.markdown("### Search Results")
                    for item in results:
                        st.markdown(f"**[{item.get('title')}]({item.get('link')})**")
                        st.write(item.get("snippet"))

def generate_search_query(context, api_key):
    prompt = f"Based on the following educational context:\n{context[:3000]}\nGenerate a concise search query to find the most relevant study resources, including helpful books, online courses, and videos."
    return call_groq_api(prompt, api_key)

if __name__ == "__main__":
    main()
