import os
import json
import requests
import pdfplumber
import streamlit as st
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Constants for local storage
SUBJECTS_DIR = "subjects"
METADATA_FILENAME = "metadata.json"
AGG_OUTPUT_FILENAME = "aggregated_output.json"

# --- Folder and Metadata Management Functions ---

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
        # Initialize metadata for subject
        save_metadata(subject, {})
        save_aggregated_outputs(subject, {})
    return subject_path

def get_metadata_path(subject):
    return os.path.join(SUBJECTS_DIR, subject, METADATA_FILENAME)

def load_metadata(subject):
    meta_path = get_metadata_path(subject)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_metadata(subject, metadata):
    meta_path = get_metadata_path(subject)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def get_agg_output_path(subject):
    return os.path.join(SUBJECTS_DIR, subject, AGG_OUTPUT_FILENAME)

def load_aggregated_outputs(subject):
    agg_path = get_agg_output_path(subject)
    if os.path.exists(agg_path):
        with open(agg_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_aggregated_outputs(subject, agg_output):
    agg_path = get_agg_output_path(subject)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_output, f, indent=2)

def save_file_to_subject(subject, uploaded_file):
    subject_path = os.path.join(SUBJECTS_DIR, subject)
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    file_path = os.path.join(subject_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- File Text Extraction Functions ---

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

def extract_text_from_uploaded(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

# --- API Call Functions ---
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
    prompt = f"Summarize the following content concisely:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_faq(text, api_key):
    prompt = f"Generate 5 FAQs with answers based on the following content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_practice_questions(text, api_key):
    prompt = f"Generate 5 practice questions with answers based on the following content:\n{text[:2000]}"
    return call_groq_api(prompt, api_key)

def generate_structured_quiz(text, api_key):
    prompt = (
        "Generate a multiple-choice quiz with 5 questions based on the following content. "
        "Return the quiz in JSON format as an array of objects. Each object should have: "
        '"question": <string>, "options": [<option A>, <option B>, <option C>, <option D>], '
        '"answer": <the letter of the correct option (A, B, C, or D) or the full text>, '
        'and "explanation": <a brief explanation of why the answer is correct>.\n'
        f"{text[:2000]}"
    )
    response_text = call_groq_api(prompt, api_key)
    # Extract JSON substring if extraneous characters exist
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

# --- Helper for Caching/Regenerating Outputs ---
def get_or_generate_output(key, generator_func, context_text, api_key, force_regen=False):
    # key: unique identifier in metadata (or aggregated metadata)
    # Returns previously stored output if exists and not forced to regenerate
    output = st.session_state.get(key)
    if not force_regen and output:
        return output
    new_output = generator_func(context_text, api_key)
    st.session_state[key] = new_output
    return new_output

# --- Main Application ---
def main():
    st.title("EduScribe: Lecture Summarizer, QnA & Chatbot")
    
    # Sidebar: Subject Folder Management
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
    
    # Sidebar: Upload File to Subject
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
            # To reduce metadata size, store only a truncated summary (or generate summary)
            if "summary" not in metadata.get(uploaded_subject_file.name, {}):
                summary = generate_summary(extracted_text, API_KEY)
            else:
                summary = metadata[uploaded_subject_file.name].get("summary")
            metadata[uploaded_subject_file.name] = {
                "filepath": file_path,
                "extracted_text": extracted_text,
                "summary": summary  # store only summary (can add FAQ, quiz etc later)
            }
            save_metadata(selected_subject, metadata)
            st.sidebar.success(f"Saved {uploaded_subject_file.name} to {selected_subject}")
    else:
        st.sidebar.info("Select or create a subject to upload files.")
    
    # Main Tab Selection
    tab = st.radio("Choose Operation", ["Folder Operations", "File Operations", "Chat"])
    
    if tab == "Folder Operations":
        if selected_subject == "--None--":
            st.info("Select a subject folder for folder operations.")
        else:
            st.header(f"Subject: {selected_subject}")
            metadata = load_metadata(selected_subject)
            # Aggregate summaries from all files to reduce context size
            aggregated_text = "\n".join([data.get("summary", "") for data in metadata.values() if data.get("summary")])
            st.subheader("Folder Aggregated Outputs")
            
            # Generate and display Aggregated Summary (stored in aggregated outputs)
            agg_outputs = load_aggregated_outputs(selected_subject)
            if st.button("Generate Aggregated Summary for Folder"):
                summary = generate_summary(aggregated_text, API_KEY)
                agg_outputs["summary"] = summary
                save_aggregated_outputs(selected_subject, agg_outputs)
            if "summary" in agg_outputs:
                st.markdown("**Folder Summary:**")
                st.write(agg_outputs["summary"])
            
            if st.button("Generate FAQs for Folder"):
                faq = generate_faq(aggregated_text, API_KEY)
                agg_outputs["faq"] = faq
                save_aggregated_outputs(selected_subject, agg_outputs)
            if "faq" in agg_outputs:
                st.markdown("**Folder FAQs:**")
                st.write(agg_outputs["faq"])
            
            if st.button("Generate Practice Questions for Folder"):
                practice = generate_practice_questions(aggregated_text, API_KEY)
                agg_outputs["practice"] = practice
                save_aggregated_outputs(selected_subject, agg_outputs)
            if "practice" in agg_outputs:
                st.markdown("**Folder Practice Questions:**")
                st.write(agg_outputs["practice"])
            
            # Aggregated Interactive Quiz with regeneration and persistent state
            if st.button("Generate Interactive Quiz for Folder") or "folder_quiz" in st.session_state:
                if st.button("Regenerate Folder Quiz"):
                    st.session_state.folder_quiz = generate_structured_quiz(aggregated_text, API_KEY)
                elif "folder_quiz" not in st.session_state:
                    st.session_state.folder_quiz = generate_structured_quiz(aggregated_text, API_KEY)
                folder_quiz = st.session_state.folder_quiz
                if folder_quiz:
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
                        # Display explanations for each question
                        for idx, q in enumerate(folder_quiz):
                            st.markdown(f"**Q{idx+1} Explanation:** {q.get('explanation', 'No explanation provided')}")
            
            st.subheader("Files in Subject")
            file_list = list(metadata.keys())
            selected_file = st.selectbox("Select a file", ["--None--"] + file_list)
            if selected_file != "--None--":
                file_data = metadata.get(selected_file, {})
                st.write(f"**File: {selected_file}**")
                st.write(file_data.get("extracted_text", "")[:500] + "...")
                # File-level outputs with regeneration options:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Summary for File"):
                        file_summary = generate_summary(file_data.get("extracted_text", ""), API_KEY)
                        file_data["summary"] = file_summary
                        metadata[selected_file] = file_data
                        save_metadata(selected_subject, metadata)
                    if "summary" in file_data:
                        st.markdown("**File Summary:**")
                        st.write(file_data["summary"])
                with col2:
                    if st.button("Generate FAQs for File"):
                        file_faq = generate_faq(file_data.get("extracted_text", ""), API_KEY)
                        file_data["faq"] = file_faq
                        metadata[selected_file] = file_data
                        save_metadata(selected_subject, metadata)
                    if "faq" in file_data:
                        st.markdown("**File FAQs:**")
                        st.write(file_data["faq"])
                if st.button("Generate Practice Questions for File"):
                    file_practice = generate_practice_questions(file_data.get("extracted_text", ""), API_KEY)
                    file_data["practice"] = file_practice
                    metadata[selected_file] = file_data
                    save_metadata(selected_subject, metadata)
                    st.markdown("**File Practice Questions:**")
                    st.write(file_practice)
                if st.button("Generate Interactive Quiz for File") or "file_quiz" in st.session_state:
                    if st.button("Regenerate File Quiz"):
                        st.session_state.file_quiz = generate_structured_quiz(file_data.get("extracted_text", ""), API_KEY)
                    elif "file_quiz" not in st.session_state:
                        st.session_state.file_quiz = generate_structured_quiz(file_data.get("extracted_text", ""), API_KEY)
                    file_quiz = st.session_state.file_quiz
                    if file_quiz:
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
            lecture_text = extract_text_from_uploaded(uploaded_file_main)
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
            if st.button("Generate Interactive Quiz for Uploaded File") or "upload_quiz" in st.session_state:
                if st.button("Regenerate Uploaded File Quiz"):
                    st.session_state.upload_quiz = generate_structured_quiz(lecture_text, API_KEY)
                elif "upload_quiz" not in st.session_state:
                    st.session_state.upload_quiz = generate_structured_quiz(lecture_text, API_KEY)
                up_quiz = st.session_state.upload_quiz
                if up_quiz:
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
                metadata = load_metadata(selected_subject)
                # Use aggregated summaries if available; otherwise, aggregate full summaries
                aggregated_context = "\n".join([data.get("summary", "") for data in metadata.values() if data.get("summary")])
                if not aggregated_context:
                    aggregated_context = "\n".join([data.get("extracted_text", "")[:1000] for data in metadata.values()])
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

if __name__ == "__main__":
    main()
