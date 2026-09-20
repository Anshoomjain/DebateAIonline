"""
FAISS Retriever - Semantic search using vector embeddings
==========================================================
CPU-based similarity search for cloud deployment.
"""

import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from core.interfaces import BaseRetriever, Document


class FAISSRetriever(BaseRetriever):
    """Semantic search using FAISS vector database."""

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        super().__init__(name="faiss")

        self.device = "cpu" 
        self.model_name = model_name

        print(f"Loading embedding model on CPU...")
        self.encoder = SentenceTransformer(model_name, device="cpu")
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        self.index = None
        self.documents: List[Document] = []

        print(f"✓ FAISSRetriever initialized (CPU mode)")
        print(f"  - Model: {model_name}")
        print(f"  - Dimension: {self.dimension}")

    def add_documents(self, documents: List[Document]):
        if not documents:
            raise ValueError("Cannot add empty document list")

        print(f"\nIndexing {len(documents)} documents...")
        self.documents = documents
        texts = [doc.text for doc in documents]

        print("Generating embeddings...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )

        # CPU FAISS index only
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"✓ Indexed {self.index.ntotal} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No documents indexed. Call add_documents() first.")

        query_embedding = self.encoder.encode(
            [query], convert_to_numpy=True
        ).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)
        similarities = 1 / (1 + distances[0])

        results = []
        for idx, score in zip(indices[0], similarities):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append(Document(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                    score=float(score)
                ))
        return results

    def save_index(self, path: str):
        if self.index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.index, path)
        print(f"✓ Saved FAISS index to {path}")

    def load_index(self, path: str, documents: List[Document]):
        self.index = faiss.read_index(path)
        self.documents = documents
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
