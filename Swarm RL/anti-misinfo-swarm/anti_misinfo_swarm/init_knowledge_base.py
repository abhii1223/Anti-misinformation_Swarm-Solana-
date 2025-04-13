"""
Initialize the knowledge base for the anti-misinformation swarm.
This script downloads and processes reliable fact-checking information 
to create a vector store for RAG.
"""

import os
import json
import logging
import yaml
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger("knowledge_base_init")
logging.basicConfig(level=logging.INFO)

# List of trusted fact-checking sites with their APIs or links
TRUSTED_SOURCES = {
    "snopes": "https://www.snopes.com/fact-check/",
    "politifact": "https://www.politifact.com/factchecks/",
    "factcheck_org": "https://www.factcheck.org/fake-news/",
    "reuters": "https://www.reuters.com/fact-check",
    "ap_factcheck": "https://apnews.com/hub/ap-fact-check",
}

# Sample facts for demonstration purposes
SAMPLE_FACTS = [
    {
        "claim": "COVID-19 vaccines can alter your DNA.",
        "verdict": "False",
        "explanation": "COVID-19 vaccines do not alter or interact with your DNA. mRNA vaccines work by instructing cells to make a protein that triggers an immune response. The mRNA does not enter the cell nucleus where DNA is kept.",
        "sources": ["CDC", "WHO", "NIH"],
        "category": "health"
    },
    {
        "claim": "5G mobile networks spread COVID-19.",
        "verdict": "False",
        "explanation": "Viruses cannot travel on radio waves/mobile networks. COVID-19 is spreading in many countries that do not have 5G mobile networks.",
        "sources": ["WHO", "Full Fact"],
        "category": "technology"
    },
    {
        "claim": "Drinking bleach can cure coronavirus.",
        "verdict": "False",
        "explanation": "Drinking bleach is extremely dangerous and can cause severe vomiting, diarrhea, and life-threatening low blood pressure. It does not cure any diseases.",
        "sources": ["CDC", "FDA"],
        "category": "health"
    },
    {
        "claim": "Earth is flat, not round.",
        "verdict": "False",
        "explanation": "Scientific evidence from multiple fields including physics, astronomy, and geology all confirm that the Earth is roughly spherical.",
        "sources": ["NASA", "National Geographic", "Scientific studies"],
        "category": "science"
    },
    {
        "claim": "Climate change is not influenced by human activities.",
        "verdict": "False",
        "explanation": "There is overwhelming scientific consensus that climate change is real and primarily caused by human activities, particularly the burning of fossil fuels.",
        "sources": ["IPCC", "NASA", "NOAA"],
        "category": "environment"
    }
]

class KnowledgeBaseInitializer:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.kb_dir = Path(self.config.get("kb_path", "knowledge_base"))
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        embedding_model = self.config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        self.model = SentenceTransformer(embedding_model)
        
        # Create FAISS index directory
        self.index_dir = self.kb_dir / "index"
        self.index_dir.mkdir(exist_ok=True)
        
        # Create raw data directory
        self.raw_dir = self.kb_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f).get("rag", {})
        else:
            # Default configuration
            config = {
                "enabled": True,
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "retrieval_top_k": 5,
                "kb_index_type": "faiss",
                "kb_path": "knowledge_base"
            }
        return config

    def initialize(self):
        """Initialize the knowledge base with fact-checking data"""
        logger.info("Initializing knowledge base...")
        
        # For demonstration, we'll use sample facts
        # In a real system, you would scrape or use APIs to fetch data from trusted sources
        facts = self._get_facts()
        
        # Process and store the facts
        self._process_facts(facts)
        
        # Build the vector index
        self._build_index(facts)
        
        logger.info("Knowledge base initialization complete!")

    def _get_facts(self):
        """Get facts from trusted sources (using sample data for this demo)"""
        logger.info("Fetching facts from trusted sources...")
        
        # In a real implementation, you would fetch data from TRUSTED_SOURCES
        # For this demo, we're using sample facts
        facts = SAMPLE_FACTS
        
        # Save raw facts
        with open(self.raw_dir / "facts.json", 'w') as f:
            json.dump(facts, f, indent=2)
            
        logger.info(f"Collected {len(facts)} facts")
        return facts

    def _process_facts(self, facts):
        """Process and structure the facts for the knowledge base"""
        logger.info("Processing facts...")
        
        processed_facts = []
        for fact in facts:
            # Create a text representation of the fact for embedding
            text_repr = f"Claim: {fact['claim']}\nVerdict: {fact['verdict']}\nExplanation: {fact['explanation']}"
            
            processed_fact = {
                "text": text_repr,
                "metadata": fact
            }
            processed_facts.append(processed_fact)
        
        # Save processed facts
        with open(self.kb_dir / "processed_facts.json", 'w') as f:
            json.dump(processed_facts, f, indent=2)
        
        return processed_facts

    def _build_index(self, facts):
        """Build a FAISS index from the facts"""
        logger.info("Building vector index...")
        
        # Create embeddings for each fact
        texts = [f"Claim: {fact['claim']}\nVerdict: {fact['verdict']}\nExplanation: {fact['explanation']}" for fact in facts]
        embeddings = self.model.encode(texts)
        
        # Normalize the embeddings
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save the index
        faiss.write_index(index, str(self.index_dir / "faiss.index"))
        
        # Save the mapping of indices to facts
        with open(self.index_dir / "index_to_fact.json", 'w') as f:
            index_map = {i: facts[i] for i in range(len(facts))}
            json.dump(index_map, f, indent=2)
        
        logger.info(f"Created FAISS index with {len(facts)} entries")

def main():
    initializer = KnowledgeBaseInitializer()
    initializer.initialize()

if __name__ == "__main__":
    main() 