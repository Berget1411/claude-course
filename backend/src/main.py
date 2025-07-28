from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json
import tempfile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import our RAG system and tools
from .rag_system import PDFRAGSystem
from .tools import AVAILABLE_TOOLS, ToolExecutor

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot", description="AI-powered document search and chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients and systems
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
rag_system = PDFRAGSystem()
tool_executor = ToolExecutor(rag_system)

# Global state for conversation history
conversation_history = []

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    use_tools: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    timestamp: str

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    upload_time: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    doc_id: Optional[str] = None
    chunk_count: Optional[int] = None

# Helper functions
def add_user_message(messages, message):
    user_message = {
        "role": "user",
        "content": message,
    }
    messages.append(user_message)

def add_assistant_message(messages, message):
    if hasattr(message, "content"):
        content_list = []
        for block in message.content:
            if block.type == "text":
                content_list.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_list.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        assistant_message = {
            "role": "assistant",
            "content": content_list,
        }
    else:
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(assistant_message)

def run_tools(message):
    """Execute tools and return results"""
    tool_requests = [
        block for block in message.content if block.type == "tool_use"
    ]
    tool_result_blocks = []

    for tool_request in tool_requests:
        try:
            tool_output = tool_executor.execute_tool(tool_request.name, tool_request.input)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": tool_output,
                "is_error": False,
            }
        except Exception as e:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks

async def generate_streaming_response(messages: List[Dict], use_tools: bool = True):
    """Generate streaming response from Claude"""
    
    # System prompt for the RAG chatbot
    system_prompt = """
    You are an intelligent document analysis assistant with access to uploaded PDF documents and library documentation.
    
    Your capabilities include:
    1. Searching through uploaded PDF documents to find relevant information
    2. Looking up library documentation using Context7
    3. Providing detailed answers based on document content
    4. Helping with coding questions by referencing both documents and library docs
    
    When answering questions:
    - Always search documents first if the question might be answered by uploaded content
    - Use library documentation tools when asked about specific frameworks or libraries
    - Provide citations and sources for your answers
    - Be concise but thorough in your responses
    - If you can't find relevant information, say so clearly
    
    Available tools:
    - search_documents: Search uploaded PDF documents
    - lookup_library_docs: Get library documentation via Context7
    - resolve_library_id: Get Context7 library IDs
    - get_document_status: See what documents are loaded
    - get_document_summary: Get summary of specific documents
    """
    
    params = {
        "model": "claude-3-7-sonnet-latest",
        "max_tokens": 6000,
        "messages": messages,
        "system": system_prompt,
    }
    
    if use_tools:
        params["tools"] = AVAILABLE_TOOLS
    
    try:
        # Main conversation loop with tool support
        while True:
            response_text = ""
            
            # Use the correct streaming API
            with client.messages.stream(**params) as stream:
                for chunk in stream:
                    if chunk.type == "text":
                        response_text += chunk.text
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk.text})}\n\n"
                    elif chunk.type == "content_block_start" and chunk.content_block.type == "tool_use":
                        yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': chunk.content_block.name})}\n\n"
                
                final_message = stream.get_final_message()
            
            add_assistant_message(messages, final_message)
            
            # Check if we need to run tools
            if final_message.stop_reason == "tool_use":
                # Run tools
                tool_results = run_tools(final_message)
                add_user_message(messages, tool_results)
                
                yield f"data: {json.dumps({'type': 'tool_complete'})}\n\n"
                
                # Continue the conversation with tool results
                params["messages"] = messages
            else:
                break
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "PDF RAG Chatbot API", "status": "running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document to the RAG system"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Add to RAG system
        doc_id = file.filename
        chunk_count = rag_system.add_pdf(temp_file_path, doc_id)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return UploadResponse(
            success=True,
            message=f"Successfully uploaded and processed {file.filename}",
            doc_id=doc_id,
            chunk_count=chunk_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Get list of uploaded documents"""
    documents = []
    
    for doc_id, doc_info in rag_system.documents.items():
        documents.append(DocumentInfo(
            doc_id=doc_id,
            filename=doc_id,
            chunk_count=len(doc_info['chunks']),
            upload_time=datetime.now().isoformat()  # Would store actual upload time in production
        ))
    
    return documents

@app.post("/chat")
async def chat_stream(message: ChatMessage):
    """Stream chat response with RAG and tool support"""
    
    global conversation_history
    
    # Add user message to history
    add_user_message(conversation_history, message.message)
    
    # Generate streaming response
    return StreamingResponse(
        generate_streaming_response(conversation_history, message.use_tools),
        media_type="text/plain"
    )

@app.post("/search")
async def search_documents(query: str = Form(...), max_results: int = Form(3)):
    """Search through uploaded documents"""
    
    if not rag_system.documents:
        raise HTTPException(status_code=404, detail="No documents uploaded")
    
    try:
        results = rag_system.search(query, k=max_results)
        return {
            "query": query,
            "results": results,
            "total_documents": len(rag_system.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the system"""
    
    if doc_id not in rag_system.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Note: In a production system, you'd want to implement proper document removal
    # For now, this is a placeholder
    return {"message": f"Document deletion not implemented yet for {doc_id}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_loaded": len(rag_system.documents),
        "total_chunks": sum(len(doc['chunks']) for doc in rag_system.documents.values()),
        "tools_available": len(AVAILABLE_TOOLS)
    }

@app.post("/clear")
async def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return {"message": "Conversation history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

