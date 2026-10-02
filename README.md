# RAG Document Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built
with Python, Streamlit, Hugging Face Sentence Transformers,
FAISS, and the Groq API.

Users can upload PDF or TXT documents and ask questions about
their contents.

## Features

- Upload PDF documents
- Upload TXT documents
- Extract document text
- Split documents into overlapping chunks
- Generate Hugging Face embeddings
- Store embeddings in FAISS
- Semantic similarity search
- Retrieve relevant document chunks
- Generate answers using Groq
- Streamlit chat interface
- Chat history
- Display retrieved source chunks
- Preserve PDF page numbers
- Secure API key handling

## How RAG Works

The application follows this pipeline:

Document
→ Text Extraction
→ Chunking
→ Embeddings
→ FAISS
→ User Question
→ Question Embedding
→ Similarity Search
→ Relevant Chunks
→ Groq
→ Answer

Instead of sending the entire document to the language model,
the application retrieves only the chunks most relevant to the
user's question.

## Technologies

- Python
- Streamlit
- Groq API
- Sentence Transformers
- FAISS
- PyPDF
- GitHub
- Streamlit Community Cloud

## Embedding Model

The project uses:

sentence-transformers/all-MiniLM-L6-v2

## Groq Model

The default model is configured inside app.py:

GROQ_MODEL = "llama-3.3-70b-versatile"

Check Groq's current supported models before deployment:

https://console.groq.com/docs/models

## Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL

Move into the project folder:

cd rag-chatbot

Install dependencies:

pip install -r requirements.txt

## Groq API Key

Create a Groq API key from:

https://console.groq.com/keys

Never place your real API key directly inside app.py.

### Local Streamlit Secret

Create:

.streamlit/secrets.toml

Add:

GROQ_API_KEY = "your-key-here"

Do not commit secrets.toml to GitHub.

## Run Locally

Run:

streamlit run app.py

Streamlit will display the local application URL.

Open it in your browser.

## Using the Application

1. Upload a PDF or TXT document.
2. Click Process Document.
3. Wait for the document to be processed.
4. Ask a question in the chat box.
5. The application retrieves relevant chunks.
6. Groq generates an answer from those chunks.
7. Expand "View retrieved context" to inspect the sources.

## Project Structure

rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

## Streamlit Community Cloud Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Create a new app.
5. Select the repository.
6. Select app.py as the entrypoint.
7. Open Advanced Settings.
8. Add the Groq API key to Secrets:

GROQ_API_KEY = "your-key-here"

9. Deploy the application.

## Security

Never commit API keys, .env files, or Streamlit secrets to
GitHub.

The .gitignore file is configured to help prevent this.

## Limitations

Scanned PDFs are not currently supported because this version
does not include OCR.

The vector database is stored in memory, so uploaded documents
must be processed again when the Streamlit session is restarted.

## Future Improvements

Possible improvements include:

- DOCX support
- OCR for scanned PDFs
- Multiple document uploads
- Persistent vector storage
- Better chunking
- Hybrid search
- Reranking
- Streaming Groq responses
- Downloadable chat history
