"""
Citation & Source Attribution System
=====================================

Tracks which sources contribute to generated answers and provides
verifiable citations with confidence scores.

Essential for:
- User trust and verification
- Academic/research applications
- Fact-checking and accountability
- Reducing hallucination risk
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import re
from collections import defaultdict


@dataclass
class Citation:
    """Represents a citation to a source document"""
    source_id: str
    source_name: str
    chunk_id: str
    chunk_text: str
    page_number: Optional[int] = None
    confidence: float = 0.0
    relevance_score: float = 0.0
    cited_text: str = ""  # Specific text that was used
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributedAnswer:
    """Answer with full source attribution"""
    answer: str
    citations: List[Citation]
    confidence_score: float
    has_sources: bool
    attribution_map: Dict[str, List[Citation]] = field(default_factory=dict)  # Sentence -> Citations


class CitationExtractor:
    """
    Extracts and tracks citations from answer generation process

    Analyzes which retrieved chunks contributed to each part of the answer.
    """

    def __init__(self):
        self.citation_style = "numbered"  # or 'inline', 'footnote'

    def extract_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        relevance_scores: List[float] = None
    ) -> AttributedAnswer:
        """
        Extract citations from generated answer

        Args:
            answer: Generated answer text
            retrieved_chunks: Chunks used for generation
            relevance_scores: Relevance score for each chunk

        Returns:
            AttributedAnswer with citations
        """
        if relevance_scores is None:
            relevance_scores = [1.0] * len(retrieved_chunks)

        citations = []
        attribution_map = defaultdict(list)

        # Split answer into sentences
        sentences = self._split_sentences(answer)

        # For each chunk, determine which parts of answer it supports
        for idx, chunk in enumerate(retrieved_chunks):
            citation = self._create_citation(
                chunk,
                idx,
                relevance_scores[idx] if idx < len(relevance_scores) else 0.5
            )

            # Find sentences that might be supported by this chunk
            supported_sentences = self._find_supported_sentences(
                sentences,
                chunk.get('content', '')
            )

            if supported_sentences:
                citations.append(citation)

                for sent in supported_sentences:
                    attribution_map[sent].append(citation)

        # Calculate overall confidence
        confidence = self._calculate_confidence(citations, len(retrieved_chunks))

        return AttributedAnswer(
            answer=answer,
            citations=citations,
            confidence_score=confidence,
            has_sources=len(citations) > 0,
            attribution_map=dict(attribution_map)
        )

    def format_answer_with_citations(
        self,
        attributed_answer: AttributedAnswer,
        style: str = "numbered"
    ) -> str:
        """
        Format answer with inline citations

        Args:
            attributed_answer: AttributedAnswer object
            style: Citation style ('numbered', 'inline', 'footnote')

        Returns:
            Formatted answer with citations
        """
        if style == "numbered":
            return self._format_numbered(attributed_answer)
        elif style == "inline":
            return self._format_inline(attributed_answer)
        elif style == "footnote":
            return self._format_footnote(attributed_answer)
        else:
            return attributed_answer.answer

    def _format_numbered(self, attributed_answer: AttributedAnswer) -> str:
        """Format with numbered citations like [1], [2]"""
        answer = attributed_answer.answer
        sentences = self._split_sentences(answer)

        formatted_parts = []
        for sent in sentences:
            # Add citations for this sentence
            if sent in attributed_answer.attribution_map:
                cites = attributed_answer.attribution_map[sent]
                cite_nums = [str(i + 1) for i, c in enumerate(attributed_answer.citations) if c in cites]
                citation_str = f" [{','.join(cite_nums)}]"
                formatted_parts.append(sent + citation_str)
            else:
                formatted_parts.append(sent)

        formatted_answer = " ".join(formatted_parts)

        # Add citation list at the end
        citation_list = "\n\nSources:\n"
        for i, citation in enumerate(attributed_answer.citations, 1):
            page_info = f", p.{citation.page_number}" if citation.page_number else ""
            confidence_info = f" (confidence: {citation.confidence:.2f})"
            citation_list += f"[{i}] {citation.source_name}{page_info}{confidence_info}\n"

        return formatted_answer + citation_list

    def _format_inline(self, attributed_answer: AttributedAnswer) -> str:
        """Format with inline citations like (Source, 2024)"""
        answer = attributed_answer.answer
        sentences = self._split_sentences(answer)

        formatted_parts = []
        for sent in sentences:
            if sent in attributed_answer.attribution_map:
                cites = attributed_answer.attribution_map[sent]
                source_names = [c.source_name for c in cites]
                citation_str = f" ({', '.join(source_names)})"
                formatted_parts.append(sent + citation_str)
            else:
                formatted_parts.append(sent)

        return " ".join(formatted_parts)

    def _format_footnote(self, attributed_answer: AttributedAnswer) -> str:
        """Format with footnote-style citations"""
        # Similar to numbered but with superscript notation
        return self._format_numbered(attributed_answer).replace("[", "<sup>").replace("]", "</sup>")

    def _create_citation(
        self,
        chunk: Dict[str, Any],
        index: int,
        relevance_score: float
    ) -> Citation:
        """Create Citation object from chunk"""
        metadata = chunk.get('metadata', {})

        return Citation(
            source_id=chunk.get('source_id', f"source_{index}"),
            source_name=chunk.get('source_name', metadata.get('file_name', f"Document {index + 1}")),
            chunk_id=chunk.get('chunk_id', f"chunk_{index}"),
            chunk_text=chunk.get('content', '')[:200],  # First 200 chars
            page_number=metadata.get('page_number'),
            confidence=relevance_score,
            relevance_score=relevance_score,
            cited_text="",
            metadata=metadata
        )

    def _find_supported_sentences(
        self,
        sentences: List[str],
        chunk_content: str
    ) -> List[str]:
        """
        Find which sentences are supported by this chunk

        Uses simple keyword overlap (can be enhanced with semantic similarity)
        """
        supported = []
        chunk_words = set(self._tokenize(chunk_content.lower()))

        for sent in sentences:
            sent_words = set(self._tokenize(sent.lower()))

            # Calculate word overlap
            overlap = len(sent_words & chunk_words)
            overlap_ratio = overlap / len(sent_words) if sent_words else 0

            # If significant overlap, consider it supported
            if overlap_ratio > 0.3:  # 30% word overlap threshold
                supported.append(sent)

        return supported

    def _calculate_confidence(self, citations: List[Citation], total_chunks: int) -> float:
        """
        Calculate overall confidence in the answer

        Based on:
        - Number of supporting sources
        - Relevance scores of sources
        - Coverage of answer by sources
        """
        if not citations:
            return 0.0

        # Average relevance of citations
        avg_relevance = sum(c.relevance_score for c in citations) / len(citations)

        # Citation coverage (how many chunks were actually used)
        coverage = len(citations) / total_chunks if total_chunks > 0 else 0

        # Combined confidence
        confidence = (avg_relevance * 0.7) + (coverage * 0.3)

        return min(confidence, 1.0)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can use nltk for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return re.findall(r'\w+', text.lower())


class SourceTracker:
    """
    Tracks which sources are most frequently cited and useful

    Helps identify:
    - Most authoritative sources
    - Most useful documents
    - Under-utilized sources
    """

    def __init__(self):
        self.citation_counts = defaultdict(int)
        self.confidence_sum = defaultdict(float)
        self.query_count = 0

    def track_citations(self, attributed_answer: AttributedAnswer):
        """Track citations from an answer"""
        self.query_count += 1

        for citation in attributed_answer.citations:
            self.citation_counts[citation.source_id] += 1
            self.confidence_sum[citation.source_id] += citation.confidence

    def get_top_sources(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most cited sources"""
        sources = []

        for source_id, count in self.citation_counts.items():
            avg_confidence = self.confidence_sum[source_id] / count if count > 0 else 0

            sources.append({
                'source_id': source_id,
                'citation_count': count,
                'avg_confidence': avg_confidence,
                'citation_rate': count / self.query_count if self.query_count > 0 else 0
            })

        # Sort by citation count
        sources.sort(key=lambda x: x['citation_count'], reverse=True)

        return sources[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get citation statistics"""
        return {
            'total_queries': self.query_count,
            'unique_sources_cited': len(self.citation_counts),
            'total_citations': sum(self.citation_counts.values()),
            'avg_citations_per_query': sum(self.citation_counts.values()) / self.query_count if self.query_count > 0 else 0
        }


class ConfidenceCalculator:
    """
    Advanced confidence calculation for attributed answers

    Considers multiple factors:
    - Source reliability
    - Semantic similarity
    - Citation density
    - Answer coherence
    """

    @staticmethod
    def calculate_advanced_confidence(
        attributed_answer: AttributedAnswer,
        source_reliability: Dict[str, float] = None
    ) -> float:
        """
        Calculate advanced confidence score

        Args:
            attributed_answer: AttributedAnswer object
            source_reliability: Optional reliability scores per source

        Returns:
            Confidence score (0-1)
        """
        if source_reliability is None:
            source_reliability = {}

        factors = []

        # Factor 1: Number of citations (more is better, up to a point)
        num_citations = len(attributed_answer.citations)
        citation_factor = min(num_citations / 5.0, 1.0)  # Normalize to 5 citations
        factors.append(citation_factor)

        # Factor 2: Average citation confidence
        if attributed_answer.citations:
            avg_confidence = sum(c.confidence for c in attributed_answer.citations) / len(attributed_answer.citations)
            factors.append(avg_confidence)

        # Factor 3: Source reliability
        if source_reliability:
            reliabilities = [source_reliability.get(c.source_id, 0.5) for c in attributed_answer.citations]
            avg_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.5
            factors.append(avg_reliability)

        # Factor 4: Citation coverage (what % of answer is cited)
        sentences = len(attributed_answer.attribution_map)
        total_sentences = len(re.split(r'[.!?]', attributed_answer.answer))
        coverage = sentences / total_sentences if total_sentences > 0 else 0
        factors.append(coverage)

        # Weighted average
        weights = [0.2, 0.3, 0.2, 0.3]  # Adjust weights as needed
        confidence = sum(f * w for f, w in zip(factors, weights[:len(factors)]))

        return min(confidence, 1.0)


if __name__ == "__main__":
    print("Citation & Source Attribution System")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ Extract citations from generated answers")
    print("✓ Track source attribution per sentence")
    print("✓ Multiple citation styles (numbered, inline, footnote)")
    print("✓ Confidence scores for citations")
    print("✓ Source usage statistics")
    print("\nUsage:")
    print("  extractor = CitationExtractor()")
    print("  attributed = extractor.extract_citations(answer, chunks)")
    print("  formatted = extractor.format_answer_with_citations(attributed)")
