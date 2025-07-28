# PDF RAG CHATBOT

Did this project as part of the Claude with the Anthropic API course
https://verify.skilljar.com/c/csufn85ji8tz

- **FastAPI** backend with Claude AI integration
- **VoyageAI** for embeddings
- **Context7** for library documentation lookup
- **React/Next.js** frontend

## Features

✅ **PDF Document Upload & Processing**

- Upload PDF documents through the web interface
- Automatic text extraction using PyMuPDF
- Intelligent chunking with contextual information
- Hybrid search (Vector + BM25) with reranking

✅ **AI-Powered Chat Interface**

- Streaming responses from Claude
- Tool-based architecture for extensibility
- Document search capabilities
- Library documentation lookup via Context7

✅ **Advanced RAG System**

- Vector embeddings with VoyageAI
- BM25 keyword search
- Reciprocal Rank Fusion (RRF) for result combination
- Claude-powered reranking for relevance

## Prerequisites

- Python 3.13+
- Node.js 18+
- Anthropic API key
- VoyageAI API key

## Setup Instructions

### 1. Backend Setup

```bash
cd backend

# Install dependencies using Poetry
poetry install

# Or use pip with the pyproject.toml
pip install -e .

# Create environment file
cp env_example.txt .env

# Edit .env with your API keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 3. Start the Backend

```bash
cd backend

# Start the FastAPI server
python src/main.py

# Or using uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Endpoints

### Document Management

- `POST /upload` - Upload PDF documents
- `GET /documents` - List uploaded documents
- `DELETE /documents/{doc_id}` - Delete a document

### Chat & Search

- `POST /chat` - Stream chat with AI (supports tools)
- `POST /search` - Direct document search
- `POST /clear` - Clear conversation history

### Health & Status

- `GET /health` - System health check
- `GET /` - API status

## Available Tools

The chatbot has access to several tools:

1. **search_documents** - Search through uploaded PDF documents
2. **lookup_library_docs** - Get library documentation via Context7
3. **resolve_library_id** - Resolve library names to Context7 IDs
4. **get_document_status** - Show loaded documents
5. **get_document_summary** - Get document summaries

## Usage Examples

### Document Upload

1. Click "Choose File" and select a PDF
2. Wait for processing (creates chunks automatically)
3. Document appears in the loaded documents list

### Chat Examples

- "What is this document about?"
- "Search for methodology section"
- "How do I use FastAPI for streaming responses?"
- "Show me examples of Anthropic SDK usage"
- "What are the main findings in the research paper?"

### Tool Usage

The AI automatically decides when to use tools based on your questions:

- Document-related questions → `search_documents`
- Library/framework questions → `lookup_library_docs`
- Status inquiries → `get_document_status`

## Architecture

### RAG System Components

1. **PDF Processing**

   - PyMuPDF for text extraction
   - Intelligent chunking by sections
   - Contextual chunk enhancement

2. **Vector Search**

   - VoyageAI embeddings (voyage-3-large)
   - Cosine similarity search
   - Configurable distance metrics

3. **Keyword Search**

   - BM25 index for exact term matching
   - Configurable parameters (k1=1.5, b=0.75)
   - Tokenization and normalization

4. **Hybrid Retrieval**

   - Reciprocal Rank Fusion (RRF)
   - Claude-powered reranking
   - Relevance scoring

5. **Context7 Integration**
   - Library documentation lookup
   - Cached responses for performance
   - Support for 25+ popular libraries

### File Structure

```
backend/
├── src/
│   ├── main.py              # FastAPI application
│   ├── rag_system.py        # Core RAG implementation
│   ├── tools.py             # Tool definitions & execution
│   └── context7_integration.py  # Context7 client wrapper
├── pyproject.toml           # Dependencies
└── env_example.txt          # Environment template

frontend/
├── app/
│   └── page.tsx             # Main React component
├── package.json             # Dependencies
└── ...                      # Next.js configuration
```
