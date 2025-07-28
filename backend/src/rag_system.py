# RAG System Components
import math
import re
import json
import random
import string
import fitz  # PyMuPDF
import voyageai
from collections import Counter
from typing import Optional, Any, List, Dict, Tuple, Protocol, Callable
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize clients
embedding_client = voyageai.Client()
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class VectorIndex:
    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn=None,
    ):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn

    def add_document(self, document: Dict[str, Any]):
        if not self._embedding_fn:
            raise ValueError(
                "Embedding function not provided during initialization."
            )
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError(
                "Document dictionary must contain a 'content' key."
            )

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not self._embedding_fn:
            raise ValueError(
                "Embedding function not provided during initialization."
            )

        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        contents = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(
                    f"Document at index {i} must contain a 'content' key."
                )
            if not isinstance(doc["content"], str):
                raise TypeError(
                    f"Document 'content' at index {i} must be a string."
                )
            contents.append(doc["content"])

        vectors = self._embedding_fn(contents)

        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def search(
        self, query: Any, k: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.vectors:
            return []

        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError(
                    "Embedding function not provided for string query."
                )
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(
            isinstance(x, (int, float)) for x in query
        ):
            query_vector = query
        else:
            raise TypeError(
                "Query must be either a string or a list of numbers."
            )

        if self._vector_dim is None:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if self._distance_metric == "cosine":
            dist_func = self._cosine_distance
        else:
            dist_func = self._euclidean_distance

        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, self.documents[i]))

        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    def add_vector(self, vector, document: Dict[str, Any]):
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError(
                "Document dictionary must contain a 'content' key."
            )

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _euclidean_distance(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False

        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _update_stats_add(self, doc_tokens: List[str]):
        self._doc_len.append(len(doc_tokens))

        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)

        self._index_built = False

    def _calculate_idf(self):
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self):
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def add_document(self, document: Dict[str, Any]):
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError(
                "Document dictionary must contain a 'content' key."
            )

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)

        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(
                    f"Document at index {i} must contain a 'content' key."
                )
            if not isinstance(doc["content"], str):
                raise TypeError(
                    f"Document 'content' at index {i} must be a string."
                )

            content = doc["content"]
            doc_tokens = self._tokenizer(content)

            self.documents.append(doc)
            self._corpus_tokens.append(doc_tokens)
            self._update_stats_add(doc_tokens)

        self._index_built = False

    def _compute_bm25_score(
        self, query_tokens: List[str], doc_index: int
    ) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            score += numerator / (denominator + 1e-9)

        return score

    def search(
        self,
        query: Any,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents:
            return []

        if isinstance(query, str):
            query_text = query
        else:
            raise TypeError("Query must be a string for BM25Index.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        raw_scores.sort(key=lambda item: item[0], reverse=True)

        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        normalized_results.sort(key=lambda item: item[1])

        return normalized_results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25VectorStore(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"


class SearchIndex(Protocol):
    def add_document(self, document: Dict[str, Any]) -> None: ...
    def add_documents(self, documents: List[Dict[str, Any]]) -> None: ...
    def search(
        self, query: Any, k: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]: ...


class Retriever:
    def __init__(
        self,
        *indexes: SearchIndex,
        reranker_fn: Optional[
            Callable[[List[Dict[str, Any]], str, int], List[str]]
        ] = None,
    ):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)
        self._reranker_fn = reranker_fn

    def add_document(self, document: Dict[str, Any]):
        if "id" not in document:
            document["id"] = "".join(
                random.choices(string.ascii_letters + string.digits, k=8)
            )

        for index in self._indexes:
            index.add_document(document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            if "id" not in doc:
                doc["id"] = "".join(
                    random.choices(string.ascii_letters + string.digits, k=8)
                )
        
        for index in self._indexes:
            index.add_documents(documents)

    def search(
        self, query_text: str, k: int = 1, k_rrf: int = 60
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        all_results = [
            index.search(query_text, k=k * 5) for index in self._indexes
        ]

        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                doc_id = id(doc)
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {
                        "doc_obj": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[doc_id]["ranks"][idx] = rank + 1

        def calc_rrf_score(ranks: List[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        scored_docs: List[Tuple[Dict[str, Any], float]] = [
            (ranks["doc_obj"], calc_rrf_score(ranks["ranks"]))
            for ranks in doc_ranks.values()
        ]

        filtered_docs = [
            (doc, score) for doc, score in scored_docs if score > 0
        ]
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        result = filtered_docs[:k]

        if self._reranker_fn is not None:
            docs_only = [doc for doc, _ in result]

            for doc in docs_only:
                if "id" not in doc:
                    doc["id"] = "".join(
                        random.choices(
                            string.ascii_letters + string.digits, k=8
                        )
                    )

            doc_lookup = {doc["id"]: doc for doc in docs_only}
            reranked_ids = self._reranker_fn(docs_only, query_text, k)

            new_result = []
            original_scores = {id(doc): score for doc, score in result}

            for doc_id in reranked_ids:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    score = original_scores.get(id(doc), 0.0)
                    new_result.append((doc, score))

            result = new_result

        return result


# Helper functions
def generate_embedding(chunks, model="voyage-3-large", input_type="document"):
    is_list = isinstance(chunks, list)
    input_data = chunks if is_list else [chunks]
    result = embedding_client.embed(input_data, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]


def reranker_fn(docs, query_text, k):
    joined_docs = "\n".join([
        f"""
        <document>
        <document_id>{doc["id"]}</document_id>
        <document_content>{doc["content"]}</document_content>
        </document>
        """
        for doc in docs
    ])

    prompt = f"""
    You are about to be given a set of documents, along with an id of each.
    Your task is to select the {k} most relevant documents to answer the user's question.

    Here is the user's question:
    <question>
    {query_text}
    </question>
    
    Here are the documents to select from:
    <documents>
    {joined_docs}
    </documents>

    Respond in the following format:
    ```json
    {{
        "document_ids": ["id1", "id2", ...] // List document ids, {k} elements long, sorted in order of decreasing relevance to the user's query.
    }}
    ```
    """

    try:
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        # Extract JSON from response
        json_start = response_text.find('```json\n') + 8
        json_end = response_text.find('\n```', json_start)
        json_str = response_text[json_start:json_end]
        
        return json.loads(json_str)["document_ids"]
    except Exception as e:
        print(f"Error in reranker: {e}")
        return [doc["id"] for doc in docs[:k]]


def chunk_by_section(document_text, chunk_size=1000):
    """Chunk document by sections or fixed size"""
    # Try to split by sections first
    pattern = r"\n## "
    sections = re.split(pattern, document_text)
    
    if len(sections) > 1:
        return sections
    
    # Fall back to fixed-size chunks
    chunks = []
    for i in range(0, len(document_text), chunk_size):
        chunks.append(document_text[i:i + chunk_size])
    
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    
    for page in doc:
        text += page.get_text() + "\n"
    
    doc.close()
    return text


def add_context(text_chunk, source_text):
    """Add contextual information to a chunk"""
    prompt = f"""
    Write a short and succinct snippet of text to situate this chunk within the 
    overall source document for the purposes of improving search retrieval of the chunk. 

    Here is the original source document:
    <document> 
    {source_text[:3000]}...
    </document> 

    Here is the chunk we want to situate within the whole document:
    <chunk> 
    {text_chunk}
    </chunk>
    
    Answer only with the succinct context and nothing else. 
    """

    try:
        response = anthropic_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        context = response.content[0].text
        return context + "\n" + text_chunk
    except Exception as e:
        print(f"Error adding context: {e}")
        return text_chunk


class PDFRAGSystem:
    def __init__(self):
        self.vector_index = VectorIndex(embedding_fn=generate_embedding)
        self.bm25_index = BM25Index()
        self.retriever = Retriever(self.bm25_index, self.vector_index, reranker_fn=reranker_fn)
        self.documents = {}
    
    def add_pdf(self, pdf_path: str, doc_id: str = None):
        """Add a PDF document to the RAG system"""
        if doc_id is None:
            doc_id = pdf_path.split("/")[-1]
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Chunk the document
        chunks = chunk_by_section(text)
        
        # Add context to chunks and index them
        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only process substantial chunks
                contextualized_chunk = add_context(chunk, text)
                contextualized_chunks.append({
                    "content": contextualized_chunk,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "original_content": chunk
                })
        
        # Add to retriever
        self.retriever.add_documents(contextualized_chunks)
        self.documents[doc_id] = {
            "original_text": text,
            "chunks": contextualized_chunks
        }
        
        return len(contextualized_chunks)
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        results = self.retriever.search(query, k=k)
        return [{"content": doc["content"], "doc_id": doc["doc_id"], "score": score} 
                for doc, score in results]
    
    def get_context_for_query(self, query: str, k: int = 3) -> str:
        """Get formatted context for a query"""
        results = self.search(query, k=k)
        context_parts = []
        
        for i, result in enumerate(results):
            context_parts.append(f"Document {i+1} (from {result['doc_id']}):\n{result['content']}")
        
        return "\n\n".join(context_parts) 