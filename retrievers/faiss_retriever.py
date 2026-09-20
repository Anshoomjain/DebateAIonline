import os
import numpy as np
from typing import List
import google.generativeai as genai
from core.interfaces import BaseRetriever, Document

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

class FAISSRetriever(BaseRetriever):
    def __init__(self, device="cpu"):
        super().__init__(name="faiss")
        self.documents: List[Document] = []
        self.embeddings = []
        print("✓ FAISSRetriever initialized (Google Embeddings)")

    def _get_embedding(self, text: str) -> List[float]:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def add_documents(self, documents: List[Document]):
        self.documents = documents
        print(f"Generating embeddings for {len(documents)} documents...")
        self.embeddings = []
        for doc in documents:
            emb = self._get_embedding(doc.text)
            self.embeddings.append(emb)
        print(f"✓ Indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_emb = self._get_embedding(query)
        
        # Cosine similarity
        scores = []
        for i, emb in enumerate(self.embeddings):
            a, b = np.array(query_emb), np.array(emb)
            score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx]
            results.append(Document(
                text=doc.text,
                source=doc.source,
                chunk_id=doc.chunk_id,
                metadata=doc.metadata,
                score=score
            ))
        return results

    def save_index(self, path: str):
        pass

    def load_index(self, path: str, documents: List[Document]):
        pass
