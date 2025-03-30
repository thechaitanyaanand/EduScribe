# EduScribe: Lecture Summarizer, QnA & Chatbot

EduScribe is an interactive web application designed to help students and educators quickly process lecture materials. The app extracts text from PDFs and text files, generates concise summaries, FAQs, practice questions, and interactive multiple-choice quizzes, and even offers a chat feature to answer user queries—all while persisting data locally. Users can organize their materials by subject folders, with each file’s metadata stored and used for aggregated folder-level operations.

## Features

- **Subject Folder Management & Persistence:**
  - Create and manage subject folders.
  - Upload lecture files (PDF or TXT) into subjects.
  - Persistently store each file’s extracted text and a concise summary in local metadata (JSON), ensuring data remains available across sessions.

- **Aggregated Folder Operations:**
  - Automatically aggregate summaries from all files within a subject.
  - Generate folder-level outputs including:
    - Aggregated Summary
    - FAQs
    - Practice Questions
    - Interactive MCQ Quiz (with answer explanations)
  - Option to regenerate each output, replacing previous results.

- **File-Level Operations:**
  - Process individual files with the same options as folders:
    - Generate summary, FAQs, practice questions, and an interactive quiz.
  - Outputs are stored locally, so previously generated results are retained until explicitly regenerated.

- **Interactive MCQ Quiz:**
  - Quizzes are generated in structured JSON format.
  - User selections persist using Streamlit’s session state.
  - Options to regenerate the quiz.
  - Quiz grading displays scores along with detailed explanations for each question.

- **Chat Feature:**
  - Chat with a document assistant using either aggregated folder data or individual file content.
  - Ask any question related to the uploaded content and receive context-specific answers.

## Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive web app interface.
- **[pdfplumber](https://github.com/jsvine/pdfplumber):** For extracting text from PDF files.
- **[Requests](https://docs.python-requests.org/):** To make API calls.
- **[python-dotenv](https://pypi.org/project/python-dotenv/):** For managing environment variables securely.
- **Local File Storage:** Files and metadata are stored locally on the user’s computer to ensure persistence.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/eduscribe.git
   cd eduscribe
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Create a `requirements.txt` file (if not provided) with the following content:

   ```
   streamlit
   pdfplumber
   requests
   python-dotenv
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the project root and add your API key:

   ```dotenv
   GROQ_API_KEY=your_actual_api_key_here
   ```

## How to Use

1. **Run the App:**

   ```bash
   streamlit run app.py
   ```

2. **Using the Interface:**
   - **Sidebar:**  
     - **Subject Management:** Create new subject folders or select an existing subject.
     - **File Upload:** Upload files (PDF or TXT) to the selected subject. The app extracts the text and generates a summary which is stored locally.
   - **Main Area Tabs:**
     - **Folder Operations:**  
       - View aggregated outputs (summary, FAQs, practice questions, interactive quiz) for all files within a subject.
       - Regenerate any output to update stored results.
       - Select individual files from the subject to perform file-level processing.
     - **File Operations:**  
       - Directly upload a file and generate outputs (summary, FAQs, practice questions, interactive quiz) without storing it in a subject folder.
     - **Chat:**  
       - Chat with a document assistant using either the aggregated folder data or a freshly uploaded file. Ask questions about the content and receive context-specific answers.

## Project Structure

```
eduscribe/
├── app.py             # Main application file
├── .env               # Environment file containing API key
├── .gitignore         # Git ignore file for temporary and sensitive files
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
└── subjects/          # Local storage folder for subject files and metadata
    ├── Subject1/
    │   ├── file1.pdf
    │   ├── metadata.json
    │   └── aggregated_output.json
    └── Subject2/
        ├── file2.txt
        ├── metadata.json
        └── aggregated_output.json
```

## Contributing

Contributions are welcome! Feel free to fork the repository and open a pull request with your enhancements. For major changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the developers of [Streamlit](https://streamlit.io/), [pdfplumber](https://github.com/jsvine/pdfplumber), and [python-dotenv](https://pypi.org/project/python-dotenv/) for their excellent tools.
- Special thanks to the API documentation for Groq which guided the implementation of AI-based outputs.
