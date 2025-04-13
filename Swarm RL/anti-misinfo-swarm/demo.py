#!/usr/bin/env python3
"""
Demo script for testing the anti-misinformation system without the full swarm.

This script provides a simple CLI interface to test the models' ability to 
detect misinformation in user-provided claims.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("misinfo_demo")

def load_api_keys(api_keys_path: str):
    """Load API keys from JSON file"""
    with open(api_keys_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Test misinformation detection without the full swarm")
    parser.add_argument("--api-keys", type=str, default="api_keys.json", help="Path to API keys file")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG to enhance analysis")
    args = parser.parse_args()
    
    try:
        # Load API keys
        api_keys = load_api_keys(args.api_keys)
        logger.info(f"Loaded API keys from {args.api_keys}")
        
        # Initialize models here instead of importing to avoid dependencies on hivemind
        from anti_misinfo_swarm.models.openai_model import OpenAIModel
        from anti_misinfo_swarm.models.perplexity_model import PerplexityModel
        
        # Check if API keys are valid
        if not api_keys.get("openai") or api_keys.get("openai") == "YOUR_OPENAI_API_KEY":
            logger.error("Invalid OpenAI API key. Please update your api_keys.json file.")
            return
            
        if not api_keys.get("perplexity") or api_keys.get("perplexity") == "YOUR_PERPLEXITY_API_KEY":
            logger.error("Invalid Perplexity API key. Please update your api_keys.json file.")
            return
        
        # Initialize models
        openai_model = OpenAIModel(api_key=api_keys["openai"])
        perplexity_model = PerplexityModel(api_key=api_keys["perplexity"])
        
        # Initialize RAG if requested
        rag_retriever = None
        if args.use_rag:
            from anti_misinfo_swarm.models.rag_retriever import RAGRetriever
            from anti_misinfo_swarm.init_knowledge_base import KnowledgeBaseInitializer
            
            # Check if knowledge base exists, if not initialize it
            kb_dir = Path("knowledge_base")
            if not kb_dir.exists() or not (kb_dir / "index" / "faiss.index").exists():
                logger.info("Knowledge base not found. Initializing...")
                initializer = KnowledgeBaseInitializer()
                initializer.initialize()
            
            rag_retriever = RAGRetriever(kb_path="knowledge_base")
            logger.info("RAG enabled for enhanced analysis")
        
        print("\n===== Anti-Misinformation Detection Demo =====")
        print("Enter claims to analyze, or type 'exit' to quit.\n")
        
        while True:
            # Get user input
            claim = input("\nEnter a claim to analyze: ")
            if claim.lower() in ['exit', 'quit', 'q']:
                break
                
            if not claim.strip():
                continue
            
            print("\nAnalyzing claim...")
            
            # Get RAG context if available
            context = []
            if rag_retriever:
                context = rag_retriever.get_context_passages(claim)
                if context:
                    print(f"Found {len(context)} relevant facts in knowledge base")
            
            # Stage 1: Analysis
            print("\n📝 Stage 1: Independent Analysis")
            print("\nOpenAI Analysis:")
            openai_analysis = openai_model.analyze_claim(claim, context)
            print(json.dumps(openai_analysis, indent=2))
            
            print("\nPerplexity Analysis:")
            perplexity_analysis = perplexity_model.analyze_claim(claim, context)
            print(json.dumps(perplexity_analysis, indent=2))
            
            # Stage 2: Critique
            print("\n🔍 Stage 2: Cross-Critique")
            print("\nOpenAI critiquing Perplexity:")
            openai_critique = openai_model.critique_analysis(perplexity_analysis, claim)
            print(json.dumps(openai_critique, indent=2))
            
            print("\nPerplexity critiquing OpenAI:")
            perplexity_critique = perplexity_model.critique_analysis(openai_analysis, claim)
            print(json.dumps(perplexity_critique, indent=2))
            
            # Stage 3: Consensus
            print("\n🤝 Stage 3: Consensus Formation")
            analyses = [openai_analysis, perplexity_analysis]
            critiques = [openai_critique, perplexity_critique]
            
            # Alternate between models for consensus to avoid bias
            if hash(claim) % 2 == 0:
                consensus = openai_model.form_consensus(analyses, critiques, claim)
                print("\nOpenAI Consensus:")
            else:
                consensus = perplexity_model.form_consensus(analyses, critiques, claim)
                print("\nPerplexity Consensus:")
                
            print(json.dumps(consensus, indent=2))
            
            # Final verdict summary
            verdict = consensus.get("final_verdict", "UNCERTAIN")
            confidence = consensus.get("confidence", 0.0)
            
            print("\n📊 Final Verdict:")
            if verdict == "MISINFORMATION":
                print(f"❌ This claim is likely MISINFORMATION (Confidence: {confidence:.2f})")
            elif verdict == "ACCURATE":
                print(f"✅ This claim is likely ACCURATE (Confidence: {confidence:.2f})")
            else:
                print(f"⚠️ This claim's accuracy is UNCERTAIN (Confidence: {confidence:.2f})")
            
            # Add a small delay before next claim
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 