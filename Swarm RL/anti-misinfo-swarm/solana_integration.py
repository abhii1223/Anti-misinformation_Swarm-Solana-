#!/usr/bin/env python3
"""
Solana integration utility for the Swarm RL misinformation detection system.
This script allows easy integration with Solana blockchain by providing methods
to get fact-checking verdicts as JSON and send them to the blockchain.
"""

import json
import argparse
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

# Import the swarm demo
from swarm_demo import get_swarm_verdict

def get_verdict_json(claim, num_peers=4, use_rag=False):
    """
    Get a verdict for a claim in JSON format ready for Solana integration
    
    Args:
        claim: The claim to fact-check
        num_peers: Number of peers to use in the swarm
        use_rag: Whether to use RAG for enhanced analysis
        
    Returns:
        JSON string with the verdict and all details
    """
    return get_swarm_verdict(claim, num_peers=num_peers, use_rag=use_rag)

def get_verdict_compact(claim, num_peers=4, use_rag=False):
    """
    Get a compact verdict for a claim (for smaller on-chain storage)
    
    Args:
        claim: The claim to fact-check
        num_peers: Number of peers to use in the swarm
        use_rag: Whether to use RAG for enhanced analysis
        
    Returns:
        Compact JSON string with just the essential verdict info
    """
    full_verdict = json.loads(get_swarm_verdict(claim, num_peers=num_peers, use_rag=use_rag))
    
    # Extract only the essential information for compact storage
    compact_verdict = {
        "claim": claim,
        "verdict": full_verdict.get("verdict", "UNCERTAIN"),
        "confidence": full_verdict.get("confidence", 0.0),
        "timestamp": full_verdict.get("timestamp", 0),
        "consensus_peer": full_verdict.get("consensus_peer", "unknown"),
        "total_peers": len(full_verdict.get("analyses_summary", [])),
        # Calculate average rewards
        "avg_reward": sum(peer.get("total_reward", 0.0) 
                         for peer in full_verdict.get("rewards", [])) / 
                     max(1, len(full_verdict.get("rewards", [])))
    }
    
    return json.dumps(compact_verdict)

def main():
    parser = argparse.ArgumentParser(description="Solana integration for Swarm RL fact-checking")
    parser.add_argument("--claim", type=str, help="Claim to fact-check")
    parser.add_argument("--num-peers", type=int, default=4, help="Number of peers to use")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for analysis")
    parser.add_argument("--compact", action="store_true", help="Output compact JSON for Solana")
    args = parser.parse_args()
    
    # If no claim provided, get it interactively
    claim = args.claim
    if not claim:
        claim = input("Enter claim to fact-check: ")
    
    # Get the verdict
    if args.compact:
        result = get_verdict_compact(claim, args.num_peers, args.use_rag)
    else:
        result = get_verdict_json(claim, args.num_peers, args.use_rag)
    
    print(result)

if __name__ == "__main__":
    main() 