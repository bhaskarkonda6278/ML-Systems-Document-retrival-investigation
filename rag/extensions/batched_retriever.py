"""Exact cosine retrieval using a matrix multiply for a batch of queries."""
from __future__ import annotations

import time
import numpy as np
from src.retriever import ExactDenseRetriever


class BatchedExactDenseRetriever(ExactDenseRetriever):
    def retrieve_many(self, query_vectors: np.ndarray, k: int = 5) -> dict:
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError("queries must have shape [batch, embedding_dimension]")
        if not np.isfinite(queries).all():
            raise ValueError("queries must be finite")
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero-norm query vector")
        if queries.shape[0] == 0:
            return {"results": [], "batch_latency_ms": 0.0}
        queries = queries / norms
        k = min(k, len(self.record_ids))
        start = time.perf_counter()
        scores = queries @ self.embeddings.T
        candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        ranked = np.take_along_axis(candidates, order, axis=1)
        elapsed = (time.perf_counter() - start) * 1000.0
        return {
            "results": [
                {"record_ids": [self.record_ids[int(i)] for i in indices],
                 "scores": [float(scores[row, i]) for i in indices]}
                for row, indices in enumerate(ranked)
            ],
            "batch_latency_ms": elapsed,
        }
