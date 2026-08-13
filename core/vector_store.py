"""
core/vector_store.py
=======================
Unified vector store interface with session-scoped isolation.
Chroma uses a per-session collection name; Pinecone uses a per-session
namespace within one shared index. Same interface either way.
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: str
    relevance_score: float = 0.0


def safe_session_key(session_id: str) -> str:
    """Sanitizes a session ID for use as both a Chroma collection name and a Pinecone namespace."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", session_id)[:63]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, min_relevance: float = 0.25) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete_all(self) -> None:
        """Wipes this session's data — used by the manual /clear endpoint."""
        ...


class ChromaVectorStore(VectorStore):
    def __init__(self, session_id: str, path: str = "./chroma_db"):
        import chromadb
        from chromadb.utils import embedding_functions

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = f"session_{safe_session_key(session_id)}"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, documents, metadatas):
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text, top_k=5, min_relevance=0.25):
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        chunks = []
        for doc, meta, dist, doc_id in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0], results["ids"][0]
        ):
            relevance = 1 - dist
            if relevance >= min_relevance:
                chunks.append(RetrievedChunk(doc, meta.get("source", "unknown"), doc_id, relevance))
        return chunks

    def delete_all(self):
        self.client.delete_collection(self.collection_name)


class PineconeVectorStore(VectorStore):
    EMBEDDING_DIM = 384

    def __init__(self, session_id: str, index_name: str = "research-assistant"):
        import time
        from pinecone import Pinecone, ServerlessSpec
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore as LangChainPinecone

        self.namespace = safe_session_key(session_id)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name

        existing = [idx["name"] for idx in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name, dimension=self.EMBEDDING_DIM, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = LangChainPinecone(
            index=self.pc.Index(index_name), embedding=self.embeddings,
            text_key="text", namespace=self.namespace,
        )

    def upsert(self, ids, documents, metadatas):
        from langchain_core.documents import Document
        docs = [Document(page_content=t, metadata=m) for t, m in zip(documents, metadatas)]
        self.store.add_documents(documents=docs, ids=ids)

    def query(self, query_text, top_k=5, min_relevance=0.25):
        results = self.store.similarity_search_with_score(query_text, k=top_k)
        chunks = []
        for doc, score in results:
            if score >= min_relevance:
                chunks.append(RetrievedChunk(
                    text=doc.page_content, source=doc.metadata.get("source", "unknown"),
                    chunk_id=doc.metadata.get("chunk_index", "unknown"), relevance_score=score,
                ))
        return chunks

    def delete_all(self):
        self.pc.Index(self.index_name).delete(delete_all=True, namespace=self.namespace)


def get_vector_store(session_id: str) -> VectorStore:
    """The single switch point. VECTOR_STORE=pinecone in production; unset locally defaults to chroma."""
    backend = os.getenv("VECTOR_STORE", "chroma").lower()
    if backend == "pinecone":
        return PineconeVectorStore(session_id=session_id)
    return ChromaVectorStore(session_id=session_id)