"""
RAG (Retrieval-Augmented Generation) component for the anti-misinformation swarm.
"""

import os
import json
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag_retriever")

class RAGRetriever:
    """Retrieval component for finding relevant knowledge context for claims."""
    
    def __init__(self, kb_path: str, embedding_model: str = "sentence-transformers/all-mpnet-base-v2", top_k: int = 5):
        """
        Initialize the RAG retriever.
        
        Args:
            kb_path: Path to the knowledge base directory
            embedding_model: Name of the embedding model to use
            top_k: Number of top results to retrieve
        """
        self.kb_path = Path(kb_path)
        self.top_k = top_k
        
        # Initialize embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Load FAISS index
        self.index_path = self.kb_path / "index"
        
        if not (self.index_path / "faiss.index").exists():
            logger.warning(f"FAISS index not found at {self.index_path / 'faiss.index'}. RAG will not work.")
            self.index = None
            self.index_map = {}
        else:
            self.index = faiss.read_index(str(self.index_path / "faiss.index"))
            
            # Load index-to-fact mapping
            with open(self.index_path / "index_to_fact.json", 'r') as f:
                self.index_map = json.load(f)
                # Convert string keys to integers
                self.index_map = {int(k): v for k, v in self.index_map.items()}
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} entries")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query to retrieve context for
            
        Returns:
            List of relevant context items with their metadata
        """
        if self.index is None:
            logger.warning("No index available, returning empty results")
            return []
        
        # Embed the query
        query_embedding = self.model.encode([query])
        
        # Normalize the embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        # Retrieve the corresponding facts
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.index_map:  # -1 means no result
                fact = self.index_map[idx]
                results.append({
                    "score": float(scores[0][i]),
                    "fact": fact
                })
        
        return results
    
    def get_context_passages(self, query: str) -> List[str]:
        """
        Get context passages for a query, formatted for inclusion in prompts.
        
        Args:
            query: Query to retrieve context for
            
        Returns:
            List of formatted context passages
        """
        retrieved_results = self.retrieve(query)
        
        passages = []
        for result in retrieved_results:
            fact = result["fact"]
            passage = f"Claim: '{fact['claim']}' - Verdict: {fact['verdict']} - {fact['explanation']}"
            passages.append(passage)
        
        return passages

def load_retriever(config: Dict[str, Any]) -> Optional[RAGRetriever]:
    """
    Load a RAG retriever from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RAG retriever or None if RAG is disabled
    """
    if not config.get("enabled", False):
        logger.info("RAG is disabled in configuration")
        return None
    
    kb_path = config.get("kb_path", "knowledge_base")
    embedding_model = config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
    top_k = int(config.get("retrieval_top_k", 5))
    
    try:
        retriever = RAGRetriever(kb_path, embedding_model, top_k)
        return retriever
    except Exception as e:
        logger.error(f"Error loading RAG retriever: {e}")
        return None 