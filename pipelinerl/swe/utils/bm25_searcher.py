import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BM25Searcher:
    """BM25 search implementation using precomputed file statistics."""
    
    def __init__(self, file_stats: Dict[str, Dict], k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 searcher with file statistics.
        
        Args:
            file_stats: Dictionary mapping file paths to their statistics
                       Format: {filepath: {"term_counts": {term: count, ...}, "length": int}}
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling length normalization
        """
        self.k1 = k1
        self.b = b
        self.file_stats = file_stats
        
        # Build inverted index and compute statistics
        self._build_index()
        
    def _build_index(self):
        """Build inverted index and compute corpus statistics."""
        self.inverted_index = defaultdict(dict)  # term -> {filepath: tf}
        self.doc_lengths = {}  # filepath -> document length
        self.doc_freqs = Counter()  # term -> number of documents containing term
        
        total_length = 0
        num_docs = 0
        
        for filepath, stats in self.file_stats.items():
            term_counts = stats.get('term_counts', {})
            doc_length = sum(term_counts.values())
            
            if doc_length == 0:
                continue  # Skip empty documents
                
            self.doc_lengths[filepath] = doc_length
            total_length += doc_length
            num_docs += 1
            
            # Add terms to inverted index
            for term, tf in term_counts.items():
                self.inverted_index[term][filepath] = tf
                
        # Compute document frequencies
        for term, docs in self.inverted_index.items():
            self.doc_freqs[term] = len(docs)
            
        # Compute average document length
        self.avg_doc_length = total_length / num_docs if num_docs > 0 else 0
        self.num_docs = num_docs
        
        logger.info(f"Built BM25 index: {num_docs} documents, {len(self.inverted_index)} unique terms")
        
    def _tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize search query using the same method as file preprocessing.
        
        Args:
            query: Search query string
            
        Returns:
            List of tokens
        """
        # Use the same tokenization as in the preprocessing
        # Split on non-alphanumeric characters (keeping underscores)
        tokens = re.findall(r'[a-zA-Z0-9_]+', query.lower())
        
        # Filter tokens (same logic as preprocessing)
        filtered_tokens = []
        for token in tokens:
            # Skip very short or very long tokens
            if len(token) < 2 or len(token) > 50:
                continue
            # Skip tokens that are all digits (unless they're meaningful like version numbers)
            if token.isdigit() and len(token) > 4:
                continue
            filtered_tokens.append(token)
            
        return filtered_tokens
        
    def _compute_bm25_score(self, query_terms: List[str], filepath: str) -> float:
        """
        Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            filepath: Path to the document
            
        Returns:
            BM25 score
        """
        if filepath not in self.doc_lengths:
            return 0.0
            
        doc_length = self.doc_lengths[filepath]
        score = 0.0
        
        # Get term frequencies for this document
        for term in query_terms:
            if term not in self.inverted_index:
                continue
                
            if filepath not in self.inverted_index[term]:
                continue
                
            tf = self.inverted_index[term][filepath]  # term frequency
            df = self.doc_freqs[term]  # document frequency
            
            # IDF component: log((N - df + 0.5) / (df + 0.5))
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5))
            
            # TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            )
            
            score += idf * tf_component
            
        return score
        
    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of (filepath, score) tuples sorted by score (descending)
        """
        if not query.strip():
            return []
            
        # Tokenize query
        query_terms = self._tokenize_query(query)
        
        if not query_terms:
            return []
            
        # Get candidate documents (documents containing at least one query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
                
        if not candidate_docs:
            return []
            
        # Compute scores for candidate documents
        scored_docs = []
        for filepath in candidate_docs:
            score = self._compute_bm25_score(query_terms, filepath)
            if score > 0:
                scored_docs.append((filepath, score))
                
        # Sort by score (descending) and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
        
    def get_stats(self) -> Dict:
        """Get statistics about the search index."""
        return {
            "num_documents": self.num_docs,
            "num_terms": len(self.inverted_index),
            "avg_doc_length": self.avg_doc_length,
            "total_doc_length": sum(self.doc_lengths.values()),
        }