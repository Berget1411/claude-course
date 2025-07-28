# Tool definitions for RAG Chatbot
from anthropic.types import ToolParam
from typing import List, Dict, Any
import json
import asyncio
from .context7_integration import context7_client

# Document search tool
search_documents_tool = ToolParam(
    {
        "name": "search_documents",
        "description": "Search through uploaded PDF documents to find relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document content"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
)

# Library documentation lookup tool
lookup_library_docs_tool = ToolParam(
    {
        "name": "lookup_library_docs",
        "description": "Look up documentation for a specific library or framework using Context7",
        "input_schema": {
            "type": "object",
            "properties": {
                "library_name": {
                    "type": "string",
                    "description": "Name of the library/framework to look up (e.g., 'fastapi', 'anthropic', 'voyageai')"
                },
                "topic": {
                    "type": "string",
                    "description": "Specific topic or feature to focus on (optional)"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens of documentation to retrieve (default: 5000)",
                    "default": 5000
                }
            },
            "required": ["library_name"]
        }
    }
)

# Context7 specific library lookup tool
resolve_library_tool = ToolParam(
    {
        "name": "resolve_library_id",
        "description": "Resolve a library name to get its Context7 compatible ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "library_name": {
                    "type": "string",
                    "description": "Library name to resolve (e.g., 'fastapi', 'anthropic')"
                }
            },
            "required": ["library_name"]
        }
    }
)

# Document upload status tool
document_status_tool = ToolParam(
    {
        "name": "get_document_status",
        "description": "Get information about currently loaded documents in the RAG system",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
)

# Get document summary tool
document_summary_tool = ToolParam(
    {
        "name": "get_document_summary",
        "description": "Get a summary of a specific document by its ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to get summary for"
                }
            },
            "required": ["doc_id"]
        }
    }
)

# All available tools
AVAILABLE_TOOLS = [
    search_documents_tool,
    lookup_library_docs_tool,
    resolve_library_tool,
    document_status_tool,
    document_summary_tool
]

# Tool execution functions
class ToolExecutor:
    def __init__(self, rag_system, context7_client=None):
        self.rag_system = rag_system
        self.context7_client = context7_client or context7_client
        self.library_cache = {}
    
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool and return the result"""
        try:
            if tool_name == "search_documents":
                return self._search_documents(**tool_input)
            elif tool_name == "lookup_library_docs":
                return self._lookup_library_docs(**tool_input)
            elif tool_name == "resolve_library_id":
                return self._resolve_library_id(**tool_input)
            elif tool_name == "get_document_status":
                return self._get_document_status()
            elif tool_name == "get_document_summary":
                return self._get_document_summary(**tool_input)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _search_documents(self, query: str, max_results: int = 3) -> str:
        """Search through uploaded documents"""
        if not self.rag_system.documents:
            return "No documents have been uploaded to search through."
        
        results = self.rag_system.search(query, k=max_results)
        
        if not results:
            return f"No relevant results found for query: {query}"
        
        response_parts = [f"Found {len(results)} relevant results for '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            response_parts.append(
                f"Result {i} (from {result['doc_id']}, relevance: {result['score']:.3f}):\n"
                f"{result['content'][:500]}{'...' if len(result['content']) > 500 else ''}\n"
            )
        
        return "\n".join(response_parts)
    
    def _lookup_library_docs(self, library_name: str, topic: str = None, max_tokens: int = 5000) -> str:
        """Look up library documentation using Context7"""
        try:
            # First resolve the library ID
            library_id = asyncio.run(context7_client.resolve_library_id(library_name))
            
            # Then get the documentation
            docs = asyncio.run(context7_client.get_library_docs(
                library_id, 
                topic=topic, 
                tokens=max_tokens
            ))
            
            return f"""
Library Documentation for {library_name}:
Library ID: {library_id}

{docs}
            """.strip()
            
        except Exception as e:
            return f"Error retrieving documentation for {library_name}: {str(e)}"
    
    def _resolve_library_id(self, library_name: str) -> str:
        """Resolve library name to Context7 ID"""
        try:
            library_id = asyncio.run(context7_client.resolve_library_id(library_name))
            return f"Resolved library ID for '{library_name}': {library_id}"
        except Exception as e:
            return f"Error resolving library ID for {library_name}: {str(e)}"
    
    def _get_document_status(self) -> str:
        """Get status of uploaded documents"""
        if not self.rag_system.documents:
            return "No documents are currently loaded in the system."
        
        status_parts = ["Current document status:\n"]
        
        for doc_id, doc_info in self.rag_system.documents.items():
            chunk_count = len(doc_info['chunks'])
            text_length = len(doc_info['original_text'])
            status_parts.append(
                f"- {doc_id}: {chunk_count} chunks, {text_length:,} characters"
            )
        
        return "\n".join(status_parts)
    
    def _get_document_summary(self, doc_id: str) -> str:
        """Get summary of a specific document"""
        if doc_id not in self.rag_system.documents:
            return f"Document '{doc_id}' not found. Available documents: {list(self.rag_system.documents.keys())}"
        
        doc_info = self.rag_system.documents[doc_id]
        text = doc_info['original_text']
        
        # Get first few paragraphs as summary
        paragraphs = text.split('\n\n')[:3]
        summary = '\n\n'.join(paragraphs)
        
        return f"Summary of {doc_id}:\n\n{summary[:1000]}{'...' if len(summary) > 1000 else ''}" 