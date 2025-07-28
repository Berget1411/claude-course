# Context7 Integration for RAG Chatbot
from typing import Optional


class Context7Client:
    """Client wrapper for Context7 library documentation lookup"""
    
    def __init__(self):
        self.library_cache = {}

    
    async def resolve_library_id(self, library_name: str) -> str:
        """
        Resolve a library name to Context7-compatible library ID
        
        This would normally call: mcp_context7_resolve-library-id
        """
        # Simulate common library mappings
        library_mappings = {
            "fastapi": "/tiangolo/fastapi",
            "anthropic": "/anthropic/anthropic-sdk-python", 
            "voyageai": "/voyageai/voyageai-python",
            "pydantic": "/pydantic/pydantic",
            "uvicorn": "/encode/uvicorn",
            "sqlalchemy": "/sqlalchemy/sqlalchemy",
            "requests": "/psf/requests",
            "numpy": "/numpy/numpy",
            "pandas": "/pandas-dev/pandas",
            "django": "/django/django",
            "flask": "/pallets/flask",
            "pytest": "/pytest-dev/pytest",
            "boto3": "/boto/boto3",
            "redis": "/redis/redis-py",
            "celery": "/celery/celery",
            "tensorflow": "/tensorflow/tensorflow",
            "pytorch": "/pytorch/pytorch",
            "scikit-learn": "/scikit-learn/scikit-learn",
            "matplotlib": "/matplotlib/matplotlib",
            "seaborn": "/mwaskom/seaborn",
            "plotly": "/plotly/plotly.py",
            "streamlit": "/streamlit/streamlit",
            "gradio": "/gradio-app/gradio",
            "huggingface": "/huggingface/transformers",
            "openai": "/openai/openai-python",
            "langchain": "/langchain-ai/langchain",
            "llama-index": "/run-llama/llama_index"
        }
        
        # Check cache first
        if library_name in self.library_cache:
            return self.library_cache[library_name]
        
        # Get library ID
        library_id = library_mappings.get(library_name.lower(), f"/{library_name.lower()}/docs")
        
        # Cache the result
        self.library_cache[library_name] = library_id
        
        return library_id
    
    async def get_library_docs(self, 
                              context7_compatible_library_id: str, 
                              topic: Optional[str] = None, 
                              tokens: int = 5000) -> str:
        """
        Get library documentation from Context7
        
        This would normally call: mcp_context7_get-library-docs
        """
        cache_key = f"{context7_compatible_library_id}:{topic}:{tokens}"
        
        if cache_key in self.library_cache:
            return self.library_cache[cache_key]
        
        # Simulate documentation retrieval
        library_name = context7_compatible_library_id.split('/')[-2] if '/' in context7_compatible_library_id else context7_compatible_library_id
        
        # Generate simulated documentation based on the library
        docs = self._generate_sample_docs(library_name, topic)
        
        # Cache the result
        self.library_cache[cache_key] = docs
        
        return docs
    
    def _generate_sample_docs(self, library_name: str, topic: Optional[str] = None) -> str:
        """Generate sample documentation for common libraries"""
        
        common_docs = {
            "fastapi": {
                "general": """
                FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

                Key features:
                - Fast: Very high performance, on par with NodeJS and Go
                - Fast to code: Increase the speed to develop features by about 200% to 300%
                - Fewer bugs: Reduce about 40% of human (developer) induced errors
                - Intuitive: Great editor support with completion everywhere
                - Easy: Designed to be easy to use and learn
                - Short: Minimize code duplication
                - Robust: Get production-ready code with automatic interactive documentation
                - Standards-based: Based on (and fully compatible with) the open standards for APIs

                Basic example:
                ```python
                from fastapi import FastAPI

                app = FastAPI()

                @app.get("/")
                async def read_root():
                    return {"Hello": "World"}

                @app.get("/items/{item_id}")
                async def read_item(item_id: int, q: str = None):
                    return {"item_id": item_id, "q": q}
                ```
                """,
                "routing": """
                FastAPI routing allows you to define API endpoints with decorators:

                ```python
                from fastapi import FastAPI

                app = FastAPI()

                @app.get("/")
                async def read_root():
                    return {"message": "Hello World"}

                @app.post("/items/")
                async def create_item(item: Item):
                    return item

                @app.get("/items/{item_id}")
                async def read_item(item_id: int, q: str = None):
                    return {"item_id": item_id, "q": q}
                ```

                Path parameters are automatically validated and converted to the correct type.
                Query parameters are also automatically parsed and validated.
                """
            },
            "anthropic": {
                "general": """
                The Anthropic Python SDK provides convenient access to the Anthropic API from Python applications.

                Installation:
                ```bash
                pip install anthropic
                ```

                Basic usage:
                ```python
                from anthropic import Anthropic

                client = Anthropic(api_key="your-api-key")

                message = client.messages.create(
                    model="claude-3-5-sonnet-20250116",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": "Hello, Claude!"}
                    ]
                )

                print(message.content)
                ```
                """,
                "streaming": """
                The Anthropic SDK supports streaming responses:

                ```python
                from anthropic import Anthropic

                client = Anthropic()

                with client.messages.stream(
                    model="claude-3-5-sonnet-20250116",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": "Tell me a story"}
                    ]
                ) as stream:
                    for chunk in stream:
                        if chunk.type == "text":
                            print(chunk.text, end="")
                ```
                """,
                "tools": """
                Claude supports function calling through tools:

                ```python
                from anthropic import Anthropic

                client = Anthropic()

                tools = [
                    {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                ]

                message = client.messages.create(
                    model="claude-3-5-sonnet-20250116",
                    max_tokens=1000,
                    tools=tools,
                    messages=[
                        {"role": "user", "content": "What's the weather in San Francisco?"}
                    ]
                )
                ```
                """
            },
            "voyageai": {
                "general": """
                Voyage AI provides state-of-the-art embedding models for retrieval and semantic search.

                Installation:
                ```bash
                pip install voyageai
                ```

                Basic usage:
                ```python
                import voyageai

                vo = voyageai.Client()

                # Single text embedding
                result = vo.embed(["Hello world"], model="voyage-3-large")
                print(result.embeddings[0])

                # Multiple texts
                texts = ["Hello world", "Goodbye world"]
                result = vo.embed(texts, model="voyage-3-large")
                print(len(result.embeddings))  # 2
                ```
                """,
                "models": """
                Voyage AI offers several embedding models:

                - voyage-3-large: Best performance, higher cost
                - voyage-3: Balanced performance and cost
                - voyage-3-lite: Faster, lower cost

                ```python
                import voyageai

                vo = voyageai.Client()

                # Use different models
                result = vo.embed(["Sample text"], model="voyage-3-large")
                result = vo.embed(["Sample text"], model="voyage-3")
                result = vo.embed(["Sample text"], model="voyage-3-lite")
                ```
                """
            }
        }
        
        if library_name in common_docs:
            if topic and topic in common_docs[library_name]:
                return common_docs[library_name][topic]
            else:
                return common_docs[library_name].get("general", f"Documentation for {library_name}")
        
        return f"""
        Documentation for {library_name}:
        
        This library provides functionality for {library_name} development.
        {'Topic: ' + topic if topic else 'General documentation'}
        
        [Note: In production, this would be retrieved from Context7's comprehensive library database]
        
        Common usage patterns and examples would be provided here along with:
        - Installation instructions
        - Basic usage examples
        - API reference
        - Best practices
        - Common patterns and recipes
        """

# Global Context7 client instance
context7_client = Context7Client() 