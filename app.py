import os
import json
import requests
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# --- Constants and Folder Management ---
SUBJECTS_DIR = "subjects"
METADATA_FILENAME = "metadata.json"

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
        save_metadata(subject_name, {})  # Create empty metadata
    return subject_path

def get_metadata_path(subject):
    return os.path.join(SUBJECTS_DIR, subject, METADATA_FILENAME)

def load_metadata(subject):
    metadata_path = get_metadata_path(subject)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_metadata(subject, metadata):
    metadata_path = get_metadata_path(subject)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def save_file_to_subject(subject, uploaded_file):
    subject_path = create_subject(subject)
    file_path = os.path.join(subject_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- File Text Extraction ---
def extract_text_from_pdf(file_path_or_buffer):
    text = ""
    # file_path_or_buffer can be a file path or an in-memory file object
    with pdfplumber.open(file_path_or_buffer) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_uploaded(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "txt":
        return uploaded_file.read().decode("utf-8")
    else:
        return ""

# --- API Calls ---
def call_groq_api(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

def generate_summary(text, api_key):
    prompt = f"Summarize the following content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_faq(text, api_key):
    prompt = f"Generate 5 frequently asked questions with answers based on the following content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_practice_questions(text, api_key):
    prompt = f"Generate 5 practice questions with answers based on the following content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_structured_quiz(text, api_key):
    prompt = (
        "Generate a multiple-choice quiz with 5 questions based on the following content. "
        "Return the quiz in JSON format as an array of objects. Each object should have "
        '"question": <string>, "options": [<option A>, <option B>, <option C>, <option D>], and '
        '"answer": <the letter of the correct option (A, B, C, or D) or the full text of the correct option>.\n'
        f"{text[:2000]}"
    )
    response_text = call_groq_api(prompt, api_key)
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
    prompt = f"Based on the following context:\n{context_text[:2000]}\nAnswer this question:\n{user_question}"
    return call_groq_api(prompt, api_key)

# --- Main Application ---
def main():
    st.title("EduScribe: Automated Lecture Summarizer, QnA & Chatbot")
    
    # Sidebar: Subject Folder Management
    st.sidebar.header("Subject Management")
    subjects = get_subjects()
    selected_subject = st.sidebar.selectbox("Select Subject", ["--None--"] + subjects)
    new_subject = st.sidebar.text_input("Create New Subject")
    if st.sidebar.button("Create Subject"):
        if new_subject.strip():
            create_subject(new_subject.strip())
            st.sidebar.success(f"Created subject: {new_subject.strip()}")
        else:
            st.sidebar.warning("Enter a valid subject name.")
    
    # Sidebar: Upload file to subject folder
    st.sidebar.subheader("Upload File to Subject")
    if selected_subject != "--None--":
        uploaded_subject_file = st.sidebar.file_uploader("Upload file for subject", key="subject_upload")
        if uploaded_subject_file:
            file_path = save_file_to_subject(selected_subject, uploaded_subject_file)
            ext = uploaded_subject_file.name.split('.')[-1].lower()
            if ext == "pdf":
                extracted_text = extract_text_from_pdf(file_path)
            elif ext == "txt":
                extracted_text = extract_text_from_txt(file_path)
            else:
                extracted_text = ""
            metadata = load_metadata(selected_subject)
            metadata[uploaded_subject_file.name] = {
                "filepath": file_path,
                "extracted_text": extracted_text
            }
            save_metadata(selected_subject, metadata)
            st.sidebar.success(f"Saved {uploaded_subject_file.name} to {selected_subject}")
    else:
        st.sidebar.info("Select or create a subject to upload files.")
    
    # Main tab selection
    tab = st.radio("Choose Operation", ["Folder Operations", "File Operations", "Chat"])
    
    if tab == "Folder Operations":
        if selected_subject == "--None--":
            st.info("Select a subject folder for folder operations.")
        else:
            st.header(f"Subject: {selected_subject}")
            metadata = load_metadata(selected_subject)
            # Aggregate extracted text from all files in the subject folder
            aggregated_text = "\n".join([data.get("extracted_text", "") for data in metadata.values()])
            
            st.subheader("Folder Summary & QnA")
            if st.button("Generate Aggregated Summary for Folder"):
                with st.spinner("Generating summary..."):
                    summary = generate_summary(aggregated_text, API_KEY)
                    st.write(summary)
            if st.button("Generate FAQs for Folder"):
                with st.spinner("Generating FAQs..."):
                    faq = generate_faq(aggregated_text, API_KEY)
                    st.write(faq)
            if st.button("Generate Practice Questions for Folder"):
                with st.spinner("Generating practice questions..."):
                    practice_q = generate_practice_questions(aggregated_text, API_KEY)
                    st.write(practice_q)
            if st.button("Generate Interactive Quiz for Folder"):
                with st.spinner("Generating quiz..."):
                    folder_quiz = generate_structured_quiz(aggregated_text, API_KEY)
                if folder_quiz:
                    st.subheader("Interactive Quiz for Folder")
                    if "folder_quiz_data" not in st.session_state:
                        st.session_state.folder_quiz_data = folder_quiz
                    for idx, q in enumerate(st.session_state.folder_quiz_data):
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
                        for idx, q in enumerate(st.session_state.folder_quiz_data):
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
                        st.success(f"Your folder quiz score: {score} out of {len(st.session_state.folder_quiz_data)}")
            
            # List files and allow file-level operations within the folder
            st.subheader("Files in Subject")
            file_list = list(metadata.keys())
            selected_file = st.selectbox("Select a file", ["--None--"] + file_list)
            if selected_file != "--None--":
                file_data = metadata.get(selected_file, {})
                file_text = file_data.get("extracted_text", "")
                st.write(f"**File: {selected_file}**")
                st.write(file_text[:500] + "...")
                if st.button("Generate Summary for File"):
                    with st.spinner("Generating summary..."):
                        file_sum = generate_summary(file_text, API_KEY)
                        st.write(file_sum)
                if st.button("Generate FAQs for File"):
                    with st.spinner("Generating FAQs..."):
                        file_faq = generate_faq(file_text, API_KEY)
                        st.write(file_faq)
                if st.button("Generate Practice Questions for File"):
                    with st.spinner("Generating practice questions..."):
                        file_practice = generate_practice_questions(file_text, API_KEY)
                        st.write(file_practice)
                if st.button("Generate Interactive Quiz for File"):
                    with st.spinner("Generating quiz..."):
                        file_quiz = generate_structured_quiz(file_text, API_KEY)
                    if file_quiz:
                        st.subheader("Interactive Quiz for File")
                        if "file_quiz_data" not in st.session_state:
                            st.session_state.file_quiz_data = file_quiz
                        for idx, q in enumerate(st.session_state.file_quiz_data):
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
                            for idx, q in enumerate(st.session_state.file_quiz_data):
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
                            st.success(f"Your file quiz score: {score} out of {len(st.session_state.file_quiz_data)}")
    
    elif tab == "File Operations":
        st.header("Direct File Upload and Processing")
        uploaded_file_main = st.file_uploader("Upload Lecture Material (PDF or TXT)", key="main_file")
        if uploaded_file_main:
            lecture_text = extract_text_from_uploaded(uploaded_file_main)
            st.write("**Extracted Text (first 500 characters):**")
            st.write(lecture_text[:500] + "...")
            if st.button("Generate Summary for Uploaded File"):
                with st.spinner("Generating summary..."):
                    up_sum = generate_summary(lecture_text, API_KEY)
                    st.write(up_sum)
            if st.button("Generate FAQs for Uploaded File"):
                with st.spinner("Generating FAQs..."):
                    up_faq = generate_faq(lecture_text, API_KEY)
                    st.write(up_faq)
            if st.button("Generate Practice Questions for Uploaded File"):
                with st.spinner("Generating practice questions..."):
                    up_practice = generate_practice_questions(lecture_text, API_KEY)
                    st.write(up_practice)
            if st.button("Generate Interactive Quiz for Uploaded File"):
                with st.spinner("Generating quiz..."):
                    up_quiz = generate_structured_quiz(lecture_text, API_KEY)
                if up_quiz:
                    st.subheader("Interactive Quiz for Uploaded File")
                    if "upload_quiz_data" not in st.session_state:
                        st.session_state.upload_quiz_data = up_quiz
                    for idx, q in enumerate(st.session_state.upload_quiz_data):
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
                        for idx, q in enumerate(st.session_state.upload_quiz_data):
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
                        st.success(f"Your uploaded file quiz score: {score} out of {len(st.session_state.upload_quiz_data)}")
    
    elif tab == "Chat":
        st.header("Chat with Document Assistant")
        chat_context = st.selectbox("Chat Context", ["Folder", "Uploaded File"])
        if chat_context == "Folder":
            if selected_subject == "--None--":
                st.info("Select a subject folder for chat.")
            else:
                metadata = load_metadata(selected_subject)
                aggregated_text = "\n".join([data.get("extracted_text", "") for data in metadata.values()])
                st.write("Aggregated context from subject folder is loaded.")
                user_question = st.text_input("Enter your question about the folder content:", key="folder_chat_question")
                if st.button("Ask Folder Question"):
                    with st.spinner("Getting response..."):
                        chat_answer = generate_chat_response(aggregated_text, user_question, API_KEY)
                        st.write("**Answer:**", chat_answer)
        elif chat_context == "Uploaded File":
            uploaded_file_chat = st.file_uploader("Upload a file for chat", key="chat_file")
            if uploaded_file_chat:
                file_ext = uploaded_file_chat.name.split('.')[-1].lower()
                if file_ext == "pdf":
                    file_text = extract_text_from_pdf(uploaded_file_chat)
                elif file_ext == "txt":
                    file_text = uploaded_file_chat.read().decode("utf-8")
                else:
                    file_text = ""
                st.write("Extracted context from uploaded file is loaded.")
                user_question = st.text_input("Enter your question about the file content:", key="file_chat_question")
                if st.button("Ask File Question"):
                    with st.spinner("Getting response..."):
                        chat_answer = generate_chat_response(file_text, user_question, API_KEY)
                        st.write("**Answer:**", chat_answer)

if __name__ == "__main__":
    main()
