"""
Hybrid Search Module
====================

Combines BM25 (keyword-based) with Dense Vector Search (semantic)
for improved retrieval accuracy.

BM25: Best for exact matches, names, dates, specific terms
Dense: Best for concepts, synonyms, semantic similarity
Fusion: Reciprocal Rank Fusion (RRF) combines both rankings

Research shows hybrid search improves retrieval by 15-25% over dense alone.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class SearchResult:
    """Single search result with score and metadata"""
    chunk_id: str
    content: str
    score: float
    source: str  # 'bm25', 'dense', or 'hybrid'
    metadata: Dict[str, Any] = None


class BM25Retriever:
    """
    BM25 (Best Matching 25) retriever for keyword-based search.

    BM25 is a probabilistic retrieval function that ranks documents
    based on query term frequency and inverse document frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.corpus_ids = []
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.initialized = False

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for BM25 search

        Args:
            documents: List of dicts with 'id', 'content', 'metadata'
        """
        self.corpus = []
        self.corpus_ids = []
        self.doc_freqs = defaultdict(int)

        # Tokenize and build corpus
        for doc in documents:
            doc_id = doc.get('id', '')
            content = doc.get('content', '')
            tokens = self._tokenize(content)

            self.corpus.append(tokens)
            self.corpus_ids.append(doc_id)
            self.doc_len.append(len(tokens))

            # Count document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Calculate IDF for each term
        num_docs = len(self.corpus)
        for term, freq in self.doc_freqs.items():
            self.idf[term] = np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)

        self.initialized = True
        print(f"[BM25] Indexed {num_docs} documents, {len(self.idf)} unique terms")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using BM25

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if not self.initialized:
            return []

        query_tokens = self._tokenize(query)
        scores = self._calculate_scores(query_tokens)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append(SearchResult(
                    chunk_id=self.corpus_ids[idx],
                    content=' '.join(self.corpus[idx][:100]),  # First 100 tokens
                    score=float(scores[idx]),
                    source='bm25',
                    metadata={'rank': len(results) + 1}
                ))

        return results

    def _calculate_scores(self, query_tokens: List[str]) -> np.ndarray:
        """Calculate BM25 scores for all documents"""
        scores = np.zeros(len(self.corpus))

        for token in query_tokens:
            if token not in self.idf:
                continue

            idf_score = self.idf[token]

            for idx, doc_tokens in enumerate(self.corpus):
                # Term frequency in document
                tf = doc_tokens.count(token)

                if tf == 0:
                    continue

                # Document length normalization
                doc_len = self.doc_len[idx]
                norm_factor = 1 - self.b + self.b * (doc_len / self.avgdl)

                # BM25 formula
                score = idf_score * (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)
                scores[idx] += score

        return scores

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (can be enhanced with nltk/spacy)"""
        import re
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens


class HybridSearchEngine:
    """
    Combines BM25 and Dense retrieval using Reciprocal Rank Fusion (RRF)

    RRF Formula: RRF(d) = Σ 1/(k + rank_i(d))
    where k is a constant (usually 60) and rank_i(d) is the rank of document d in list i
    """

    def __init__(self, bm25_weight: float = 0.6, dense_weight: float = 0.4, rrf_k: int = 60):
        """
        Initialize hybrid search engine

        Args:
            bm25_weight: Weight for BM25 results (0-1)
            dense_weight: Weight for dense results (0-1)
            rrf_k: RRF constant (default: 60)
        """
        self.bm25_retriever = BM25Retriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        print(f"[Hybrid Search] Initialized (BM25: {bm25_weight}, Dense: {dense_weight})")

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 search"""
        self.bm25_retriever.index_documents(documents)

    def hybrid_search(
        self,
        query: str,
        dense_results: List[SearchResult],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and dense results

        Args:
            query: Search query
            dense_results: Results from dense/vector search
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        # Get BM25 results
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        # Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k
        )

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion

        RRF gives higher scores to documents that appear in multiple rankings,
        especially if they appear near the top.
        """
        # Build score dictionary
        scores = defaultdict(float)
        content_map = {}
        metadata_map = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            scores[result.chunk_id] += rrf_score
            content_map[result.chunk_id] = result.content
            metadata_map[result.chunk_id] = result.metadata or {}
            metadata_map[result.chunk_id]['bm25_rank'] = rank

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            scores[result.chunk_id] += rrf_score

            if result.chunk_id not in content_map:
                content_map[result.chunk_id] = result.content
                metadata_map[result.chunk_id] = result.metadata or {}

            metadata_map[result.chunk_id]['dense_rank'] = rank

        # Sort by combined score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Create fused results
        fused_results = []
        for chunk_id, score in sorted_ids:
            metadata = metadata_map[chunk_id]
            metadata['fusion_score'] = score

            # Determine source
            has_bm25 = 'bm25_rank' in metadata
            has_dense = 'dense_rank' in metadata
            if has_bm25 and has_dense:
                source = 'hybrid'
            elif has_bm25:
                source = 'bm25'
            else:
                source = 'dense'

            fused_results.append(SearchResult(
                chunk_id=chunk_id,
                content=content_map[chunk_id],
                score=score,
                source=source,
                metadata=metadata
            ))

        return fused_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get hybrid search statistics"""
        return {
            'bm25_indexed': self.bm25_retriever.initialized,
            'bm25_docs': len(self.bm25_retriever.corpus),
            'bm25_vocab_size': len(self.bm25_retriever.idf),
            'bm25_weight': self.bm25_weight,
            'dense_weight': self.dense_weight,
            'rrf_k': self.rrf_k
        }


class WeightedFusion:
    """
    Alternative fusion strategy using weighted scores

    Simpler than RRF but requires score normalization
    """

    @staticmethod
    def fuse(
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        bm25_weight: float = 0.5,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Fuse results using weighted average of normalized scores

        Args:
            bm25_results: BM25 search results
            dense_results: Dense search results
            bm25_weight: Weight for BM25 (dense gets 1 - bm25_weight)
            top_k: Number of results to return
        """
        # Normalize scores to 0-1 range
        bm25_scores = [r.score for r in bm25_results]
        dense_scores = [r.score for r in dense_results]

        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        dense_max = max(dense_scores) if dense_scores else 1.0

        # Combine scores
        combined = defaultdict(float)
        content_map = {}

        for result in bm25_results:
            norm_score = result.score / bm25_max
            combined[result.chunk_id] += norm_score * bm25_weight
            content_map[result.chunk_id] = result.content

        for result in dense_results:
            norm_score = result.score / dense_max
            combined[result.chunk_id] += norm_score * (1 - bm25_weight)
            if result.chunk_id not in content_map:
                content_map[result.chunk_id] = result.content

        # Sort and return top-k
        sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(
                chunk_id=chunk_id,
                content=content_map[chunk_id],
                score=score,
                source='hybrid_weighted'
            )
            for chunk_id, score in sorted_ids
        ]


def create_hybrid_search(bm25_weight: float = 0.6, dense_weight: float = 0.4, rrf_k: int = 60) -> HybridSearchEngine:
    """
    Factory function to create HybridSearchEngine
    
    Args:
        bm25_weight: Weight for BM25 results (0-1)
        dense_weight: Weight for dense results (0-1)
        rrf_k: RRF constant (default: 60)
        
    Returns:
        Initialized HybridSearchEngine
    """
    return HybridSearchEngine(bm25_weight, dense_weight, rrf_k)


if __name__ == "__main__":
    print("Hybrid Search Module")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ BM25 keyword-based retrieval")
    print("✓ Dense semantic retrieval")
    print("✓ Reciprocal Rank Fusion (RRF)")
    print("✓ Weighted fusion alternative")
    print("\nUsage:")
    print("  engine = HybridSearchEngine()")
    print("  engine.index_documents(documents)")
    print("  results = engine.hybrid_search(query, dense_results)")
