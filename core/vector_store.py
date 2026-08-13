# core/vector_store.py
"""
Unified vector store interface — swap ChromaDB (local dev) for Pinecone
(production) via one environment variable, with zero changes needed in
any phase's retrieval code.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: str
    relevance_score: float = 0.0


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None: ...

    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, min_relevance: float = 0.25) -> list[RetrievedChunk]: ...


class ChromaVectorStore(VectorStore):
    """Local, file-based — used for all development, testing, and evaluation."""

    def __init__(self, path: str = "./chroma_db", collection_name: str = "research_assistant_docs"):
        import chromadb
        from chromadb.utils import embedding_functions

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=path)
        self.collection = client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn, metadata={"hnsw:space": "cosine"}
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


"""
core/vector_store.py — Pinecone backend, rewritten to use LangChain
=======================================================================
Uses langchain-pinecone's vector store wrapper and langchain-huggingface's
embeddings wrapper — genuine LangChain usage, not a hand-rolled equivalent.
"""

import os
import time


class PineconeVectorStore(VectorStore):
    """Cloud-hosted, persistent across serverless deploys — built on LangChain."""

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension — must match the Pinecone index

    def __init__(self, index_name: str = "research-assistant", namespace: str = "default"):
        from pinecone import Pinecone, ServerlessSpec
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore as LangChainPinecone

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Create the index if it doesn't exist yet — idempotent, safe to run every time
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=self.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free tier region
            )
            # Index creation is async — wait until it's ready before using it
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = LangChainPinecone(
            index=pc.Index(index_name),
            embedding=self.embeddings,
            text_key="text",
        )
        
        self.namespace = namespace
        self.store = LangChainPinecone(
            index=pc.Index(index_name),
            embedding=self.embeddings,
            text_key="text",
            namespace=self.namespace,   # <-- scopes every upsert AND query
        )


    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        from langchain_core.documents import Document
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(documents, metadatas)
        ]
        self.store.add_documents(documents=docs, ids=ids)

    def query(self, query_text: str, top_k: int = 5, min_relevance: float = 0.25) -> list[RetrievedChunk]:
        results = self.store.similarity_search_with_score(query_text, k=top_k)
        chunks = []
        for doc, score in results:
            if score >= min_relevance:
                chunks.append(RetrievedChunk(
                    text=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    chunk_id=doc.metadata.get("chunk_index", "unknown"),
                    relevance_score=score,
                ))
        return chunks

def get_vector_store() -> VectorStore:
    """
    The single switch point. Set VECTOR_STORE=pinecone in production's .env;
    leave unset (defaults to chroma) for all local development and testing.
    """
    backend = os.getenv("VECTOR_STORE", "chroma").lower()
    if backend == "pinecone":
        return PineconeVectorStore()
    return ChromaVectorStore()