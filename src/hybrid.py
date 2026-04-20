import numpy as np
from rank_bm25 import BM25Okapi
from src.vectorstore import search as vector_search


class HybridSearcher:
    def __init__(self, chunks):
        self.chunks = chunks
        self.corpus = [c["content"] for c in chunks]
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5):
        # Vector search
        vec_results = vector_search(query, top_k=20)

        # BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25_idx = np.argsort(bm25_scores)[-20:][::-1]

        # Reciprocal Rank Fusion
        scores = {}

        for rank, r in enumerate(vec_results):
            key = r["content"][:100]
            scores[key] = {"score": 1 / (rank + 60), "data": r}

        for rank, idx in enumerate(top_bm25_idx):
            key = self.corpus[idx][:100]
            if key in scores:
                scores[key]["score"] += 1 / (rank + 60)
            else:
                scores[key] = {
                    "score": 1 / (rank + 60),
                    "data": {
                        "content": self.corpus[idx],
                        "metadata": self.chunks[idx]["metadata"],
                        "similarity": 0
                    }
                }

        ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return [r["data"] for r in ranked]
