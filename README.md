# Automated Lecture Summarizer & Quiz Generator

This project is an interactive web application designed to help students quickly digest lecture content. It extracts text from lecture materials (PDF or plain text) and uses an AI API to generate a concise summary, frequently asked questions (FAQs), practice questions, and a multiple-choice quiz.

## Features

- **File Upload:** Accepts lecture materials in PDF and text formats.
- **Content Extraction:** Uses `pdfplumber` to extract text from PDFs.
- **AI Processing:** 
  - Generates a summary of the lecture.
  - Produces a list of FAQs with answers.
  - Creates practice questions with answers.
  - Generates a multiple-choice quiz.
- **Environment Management:** Securely manages the API key using a `.env` file with `python-dotenv`.

## Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive web app.
- **[pdfplumber](https://github.com/jsvine/pdfplumber):** To extract text from PDF files.
- **[Requests](https://docs.python-requests.org/):** To make HTTP requests to the AI API.
- **[python-dotenv](https://pypi.org/project/python-dotenv/):** To load environment variables from a `.env` file.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Create a `requirements.txt` file with the following content:

   ```
   streamlit
   pdfplumber
   requests
   python-dotenv
   ```

   Then install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the project root and add your Groq API key:

   ```dotenv
   GROQ_API_KEY=your_actual_api_key_here
   ```

## Usage

Run the app with Streamlit:

```bash
streamlit run app.py
```

Once the app launches in your browser:
- Upload your lecture material (PDF or text).
- View a preview of the extracted text.
- Click the buttons to generate a summary, FAQs, practice questions, or a quiz.

## Project Structure

```
your-repo-name/
├── app.py         # Main application file
├── .env           # Environment variables (API key)
├── .gitignore     # Git ignore rules
├── README.md      # Project documentation
└── requirements.txt  # Python dependencies
```

## Contributing

Contributions are welcome! Please fork this repository, create a new branch for your changes, and open a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the developers of [Streamlit](https://streamlit.io/), [pdfplumber](https://github.com/jsvine/pdfplumber), and [python-dotenv](https://pypi.org/project/python-dotenv/) for their amazing tools.
- API documentation for the Groq API provided valuable guidance in building this project.
```