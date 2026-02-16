# RAGBot

An AI-powered chatbot that answers questions about specific reference books using Retrieval-Augmented Generation (RAG). Users select a book, ask questions, and receive contextual answers synthesized from relevant passages.

## Supported Books

- **Deep Learning with Python**
- **Python Data Science Handbook**

## Tech Stack

- **Backend**: Flask (Python 3.12)
- **LLM**: Google Gemini 2.5-Flash
- **Embeddings**: Voyage AI (`voyage-3-lite`)
- **Vector DB**: Qdrant (cloud)
- **Frontend**: HTML/CSS/JavaScript with Marked.js for markdown rendering
- **Deployment**: Vercel (serverless) / Heroku

## How It Works

1. User selects a reference book
2. User submits a question
3. The query is expanded with the LLM to capture synonyms and related terms
4. An embedding is generated and used to search Qdrant for the top 5 matching text chunks
5. The retrieved context and original question are passed to the LLM to generate a final answer

## Project Structure

```
ragbot-backend/
├── api/
│   └── index.py              # Vercel serverless entry point
├── public/
│   ├── index.html             # Frontend UI
│   ├── script.js              # Frontend logic
│   └── style.css              # Styling
├── app.py                     # Flask app (local development)
├── base_rag2.py               # Core RAG pipeline (Embedd, LLM, load_db, rag_process)
├── create_pickle_simple.py    # Generate embeddings from PDFs
├── qdrant_save.py             # Load embeddings into Qdrant
├── data_dl.pkl                # Pre-computed embeddings (Deep Learning)
├── data_ds.pkl                # Pre-computed embeddings (Data Science)
├── vercel.json                # Vercel deployment config
├── Procfile                   # Heroku deployment config
└── requirements.txt           # Python dependencies
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/select_book` | Select a reference book |
| GET | `/status` | Check if the system is ready |
| POST | `/ask` | Submit a question |
| POST | `/end_session` | End the current session |

## Setup

### Prerequisites

- Python 3.12
- API keys for Google Gemini, Voyage AI, and Qdrant

### Environment Variables

```
GOOGLE_API_KEY=<your-google-api-key>
VOYAGE_API_KEY=<your-voyage-api-key>
QDRANT_URL=<your-qdrant-cloud-url>
QDRANT_API_KEY=<your-qdrant-api-key>
```

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

### Deploy to Vercel

```bash
vercel
```

## Data Pipeline

To add a new book:

1. Place the PDF in the project directory
2. Run `create_pickle_simple.py` to chunk the text and generate embeddings
3. Run `qdrant_save.py` to upload the embeddings to Qdrant
4. Add the new book name and collection mapping in `base_rag2.py` and the API layer
