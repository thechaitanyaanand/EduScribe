# EduScribe RAG: Lecture Summarizer, QnA & Resource Finder

EduScribe RAG is an interactive web application that leverages Retrieval-Augmented Generation (RAG) techniques to help students and educators process and study large volumes of educational material. It extracts text from PDFs and text files (even entire books), uses hierarchical summarization and document chunking to reduce input tokens while enhancing output quality, and stores metadata persistently using SQLite. Additionally, EduScribe RAG integrates FAISS for efficient vector retrieval and offers a SERP API–powered web search feature to discover relevant study resources (books, videos, courses). A chat interface further allows users to ask context-specific questions based on aggregated folder or file content.

# Demo Video Link:
- https://drive.google.com/drive/folders/1Ybb6Xm1L4NXcylg_-XsigLxJV6t8MwOl?usp=drive_link

## Features

- **Subject Folder Management & Persistence:**
  - Create and manage subject folders.
  - Upload lecture files (PDF or TXT) into subjects.
  - Persistently store each file’s extracted text, hierarchical summaries, and document chunks (with vector embeddings) in a local SQLite database.

- **Hierarchical Summarization & Document Chunking:**
  - Automatically split long texts into overlapping chunks.
  - Summarize each chunk and then aggregate these summaries for a final concise summary.
  - Drastically reduce the total input tokens sent to the API while preserving context.

- **FAISS Vector Indexing:**
  - Compute vector embeddings for document chunks using SentenceTransformer.
  - Store and update a global FAISS index to enable efficient similarity search and retrieval.

- **Aggregated Folder Operations:**
  - Aggregate summaries from all files in a subject.
  - Generate folder-level outputs such as:
    - Aggregated Summary
    - FAQs
    - Practice Questions
    - Interactive MCQ Quiz (with detailed explanations)
  - Options to regenerate outputs, which update stored results.

- **File-Level Operations:**
  - Process individual files with the same options (summary, FAQs, practice questions, interactive quiz).
  - Previously generated outputs persist until explicitly regenerated.

- **Interactive MCQ Quiz:**
  - Quizzes are generated in structured JSON format.
  - User selections persist using Streamlit’s session state.
  - Quiz grading displays scores and detailed explanations for each question.

- **Chat Feature:**
  - Chat with a document assistant using either aggregated folder context or an individual file’s content.
  - Receive context-specific answers based on the provided study material.

- **Web Search Integration:**
  - Leverage the SERP API to automatically generate a search query from subject or file context.
  - Retrieve and display relevant study resources such as books, videos, and online courses.

## Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive web app interface.
- **[pdfplumber](https://github.com/jsvine/pdfplumber):** For extracting text from PDF files.
- **[Requests](https://docs.python-requests.org/):** To make API calls.
- **[python-dotenv](https://pypi.org/project/python-dotenv/):** For secure environment variable management.
- **[SentenceTransformer](https://www.sbert.net/):** For computing text embeddings.
- **[FAISS](https://github.com/facebookresearch/faiss):** For efficient vector similarity search.
- **SQLite:** For persistent local storage of file metadata and summaries.
- **SERP API:** For web search to find additional study resources.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/thechaitanyaanand/eduscribe.git
   cd eduscribe
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Create a `requirements.txt` with:

   ```
   streamlit
   pdfplumber
   requests
   python-dotenv
   sentence-transformers
   faiss-cpu
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the project root and add your API keys:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key_here
   SERP_API_KEY=your_serp_api_key_here
   ```

## How to Use

1. **Run the App:**

   ```bash
   streamlit run app.py
   ```

2. **Using the Interface:**
   - **Sidebar:**
     - **Subject Management:** Create new subject folders or select an existing one.
     - **File Upload:** Upload lecture files (PDF or TXT). The app extracts full text, generates a hierarchical summary, and chunks the document for efficient retrieval. Metadata is stored persistently.
   - **Main Area Tabs:**
     - **Folder Operations:**
       - View aggregated folder outputs (summary, FAQs, practice questions, interactive quiz).
       - Regenerate outputs to update stored results.
       - Select individual files within the subject to process them at the file level.
     - **File Operations:**
       - Directly upload a file and generate outputs without associating it with a subject.
     - **Chat:**
       - Chat with a document assistant using either aggregated folder context or individual file content.
     - **Web Search:**
       - Choose a context source (manual query, subject context, or uploaded file context) to automatically generate a search query.
       - Retrieve and display relevant study resources (books, videos, courses) using the SERP API.

## Project Structure

```
eduscribe/
├── app.py             # Main application file
├── .env               # Environment file containing API keys
├── .gitignore         # Git ignore file for temporary and sensitive files
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── metadata.db        # SQLite database for persistent metadata
├── faiss.index        # FAISS index file for vector embeddings
├── faiss_mapping.json # Mapping for FAISS index metadata
└── subjects/          # Local storage folder for subject files and metadata
    ├── Subject1/
    │   ├── file1.pdf
    │   └── ... (metadata stored in SQLite)
    └── Subject2/
        ├── file2.txt
        └── ... (metadata stored in SQLite)
```

## Contributing

Contributions are welcome! Feel free to fork the repository and open a pull request with your improvements. For major changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the developers of [Streamlit](https://streamlit.io/), [pdfplumber](https://github.com/jsvine/pdfplumber), [python-dotenv](https://pypi.org/project/python-dotenv/), and [SentenceTransformer](https://www.sbert.net/) for their excellent tools.
- Special thanks to the API documentation for Groq and SERP API which guided the implementation of AI-based outputs and web search functionality.
