# EDIA Backend

This repository contains the backend for Enterprise Document Intelligence Assistant (EDIA), a sophisticated RAG (Retrieval-Augmented Generation) system designed to enable intelligent querying of enterprise documents.

## Overview

EDIA is a document intelligence platform that allows users to upload PDF documents and then ask questions about their content. The system leverages advanced AI techniques to understand and retrieve relevant information from the documents, providing accurate answers with source citations.

### Key Features
- **Document Upload**: Support for PDF document uploads
- **Intelligent Search**: Semantic search capabilities using vector embeddings
- **RAG Pipeline**: Retrieval-Augmented Generation for accurate answers
- **Source Attribution**: Citations showing which documents and pages informed the response
- **Scalable Architecture**: Built with FastAPI for high-performance API endpoints

## Architecture

The system follows a modular architecture with the following components:

### 1. API Layer
- **FastAPI**: Modern, fast web framework for building APIs
- **CORS Middleware**: Handles cross-origin resource sharing
- **RESTful Endpoints**: Clean API design for document management and querying

### 2. Document Processing Pipeline
- **PDF Parser**: Extracts text content from PDF documents
- **Text Cleaner**: Normalizes and cleans extracted text
- **Chunking Algorithm**: Splits documents into manageable chunks with overlap for context preservation

### 3. Embedding Service
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` model for generating text embeddings
- **Semantic Understanding**: Converts text to high-dimensional vectors representing meaning

### 4. Vector Storage
- **ChromaDB**: Persistent vector database for efficient similarity search
- **Indexing**: Stores document embeddings with metadata for retrieval
- **Similarity Search**: Finds semantically similar document chunks to user queries

### 5. RAG Engine
- **OpenAI Integration**: Uses GPT-4o-mini for answer generation
- **Prompt Engineering**: Constructs context-aware prompts with retrieved documents
- **Source Extraction**: Identifies and returns source documents for transparency

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service.

### Document Upload
```
POST /upload
```
Uploads a PDF document and indexes it for querying.
- **Request**: Multipart form data with PDF file
- **Response**: Success message with document statistics

### Document Query
```
POST /query
```
Queries the indexed documents and returns answers with sources.
- **Request Body**:
  ```json
  {
    "query": "Your question here",
    "top_k": 5
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Generated answer based on documents",
    "sources": [
      {
        "documentName": "filename.pdf",
        "pageNumber": 1,
        "text": "Relevant text excerpt"
      }
    ]
  }
  ```

## Technology Stack

- **Python 3.8+**: Core programming language
- **FastAPI**: Web framework with automatic API documentation
- **PyPDF2**: PDF parsing and text extraction
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **ChromaDB**: Vector database for similarity search
- **OpenAI API**: Language model for answer generation
- **LangChain**: Framework for developing LLM applications
- **FAISS**: Efficient similarity search and clustering
- **Pydantic**: Data validation and settings management

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd edia-backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8000
```

2. Access the API documentation at `http://localhost:8000/docs`

## Usage

### Uploading Documents
1. Navigate to the `/docs` endpoint to access the API documentation
2. Use the `/upload` endpoint to upload PDF documents
3. The system will process and index the documents automatically

### Querying Documents
1. Use the `/query` endpoint to ask questions about your documents
2. The system will return answers with citations to the source documents

## Project Structure

```
edia-backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration settings
│   ├── constants.py            # Application constants
│   ├── api/
│   │   ├── routes.py           # API route definitions
│   │   └── schemas.py          # Pydantic models for request/response
│   ├── services/
│   │   ├── ingestion.py        # Document processing pipeline
│   │   ├── embeddings.py       # Embedding generation service
│   │   ├── vector_store.py     # Vector database operations
│   │   ├── rag.py              # RAG pipeline implementation
│   │   └── retriever.py        # Document retrieval logic
│   └── utils/
│       ├── pdf_parser.py       # PDF text extraction
│       ├── text_cleaner.py     # Text normalization
│       ├── logger.py           # Logging configuration
│       └── errors.py           # Custom error handling
├── storage/
│   ├── documents/              # Uploaded PDF documents
│   └── vectordb/               # ChromaDB vector storage
├── tests/
│   └── test_rag.py             # Unit and integration tests
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .env.example               # Environment variables template
```

## Development

### Testing
Run the test suite:
```bash
pytest tests/
```

### Code Quality
The project follows PEP 8 coding standards. Consider using linters like flake8 or black for code formatting.

## Security Considerations

- API keys should be stored securely in environment variables
- Input validation is performed on file uploads
- CORS is configured to restrict cross-origin requests
- File type validation ensures only PDFs are processed

## Performance Considerations

- Document chunking with overlap preserves context while maintaining efficiency
- Vector database enables fast similarity search
- Embedding caching could be implemented for improved performance
- Asynchronous processing for large documents

## Troubleshooting

### Common Issues
- **OpenAI API Key**: Ensure your API key is correctly set in the environment
- **PDF Format**: Verify that uploaded PDFs are not password-protected or corrupted
- **Memory Usage**: Large documents may require significant memory for processing

### Logging
The application uses Python's logging module to record important events and errors. Check logs for troubleshooting information.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Contact

For questions or support, please contact [muhammadali35484@gmail.com].