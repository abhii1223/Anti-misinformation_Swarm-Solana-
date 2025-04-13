#!/usr/bin/env python3
"""
Enhanced demo script for visualizing the Swarm RL architecture for misinformation detection.

This script simulates multiple peers in the swarm, shows rewards, and demonstrates 
the three-stage reinforcement learning process with user queries.
"""

import os
import json
import time
import random
import argparse
import logging
import colorama
from colorama import Fore, Style
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("swarm_demo")

# Initialize colorama
colorama.init()

def load_api_keys(api_keys_path: str):
    """Load API keys from JSON file"""
    with open(api_keys_path, 'r') as f:
        return json.load(f)

class SwarmPeer:
    """Simulated swarm peer for demo purposes"""
    
    def __init__(self, name: str, model_type: str, api_key: str, context_provider=None):
        self.name = name
        self.model_type = model_type
        self.api_key = api_key
        self.context_provider = context_provider
        self.rewards = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0, "total": 0.0}
        
        # Initialize the appropriate model
        if model_type == "openai":
            from anti_misinfo_swarm.models.openai_model import OpenAIModel
            self.model = OpenAIModel(api_key=api_key)
        elif model_type == "perplexity":
            from anti_misinfo_swarm.models.perplexity_model import PerplexityModel
            self.model = PerplexityModel(api_key=api_key)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def analyze_claim(self, claim: str, context: List[str] = None) -> Dict[str, Any]:
        """Run stage 1: Analyze a claim"""
        analysis = self.model.analyze_claim(claim, context)
        
        # Calculate and add reward
        reward = self._calculate_stage1_reward(analysis)
        analysis["reward"] = reward
        self.rewards["stage1"] += reward
        self.rewards["total"] += reward
        
        return analysis
    
    def critique_analysis(self, analysis: Dict[str, Any], claim: str) -> Dict[str, Any]:
        """Run stage 2: Critique an analysis"""
        critique = self.model.critique_analysis(analysis, claim)
        
        # Calculate and add reward
        reward = self._calculate_stage2_reward(critique, analysis)
        critique["reward"] = reward
        critique["peer_name"] = self.name  # Add the peer name for JSON output
        self.rewards["stage2"] += reward
        self.rewards["total"] += reward
        
        return critique
    
    def form_consensus(self, analyses: List[Dict[str, Any]], critiques: List[Dict[str, Any]], claim: str) -> Dict[str, Any]:
        """Run stage 3: Form consensus"""
        consensus = self.model.form_consensus(analyses, critiques, claim)
        
        # Calculate and add reward
        reward = self._calculate_stage3_reward(consensus, analyses)
        consensus["reward"] = reward
        consensus["peer_name"] = self.name  # Add the peer name for JSON output
        self.rewards["stage3"] += reward
        self.rewards["total"] += reward
        
        return consensus
    
    def _calculate_stage1_reward(self, analysis: Dict[str, Any]) -> float:
        """Calculate reward for a stage 1 analysis"""
        reward = 0.0
        
        # Check confidence
        confidence = float(analysis.get("confidence", 0.0))
        
        # Check quality of reasoning
        reasoning = analysis.get("reasoning", "")
        reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
        evidence = analysis.get("evidence", [])
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        
        # Basic reward for participation
        reward += 0.1
        
        # Reward for detailed reasoning (up to 0.4)
        reward += min(0.4, reasoning_length / 1000)
        
        # Reward for providing evidence (up to 0.3)
        reward += min(0.3, evidence_count * 0.1)
        
        # Reward for appropriate confidence (up to 0.2)
        if confidence > 0.0 and confidence <= 1.0:
            reward += 0.2 * confidence
        
        return reward * 0.4  # Stage 1 weight: 40%
    
    def _calculate_stage2_reward(self, critique: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Calculate reward for a stage 2 critique"""
        reward = 0.0
        
        # Check quality of critique points
        critique_points = critique.get("critique_points", [])
        points_count = len(critique_points) if isinstance(critique_points, list) else 0
        
        # Check if score and recommendations are provided
        has_score = "score" in critique and isinstance(critique["score"], (int, float))
        has_recommendations = "recommendations" in critique and isinstance(critique["recommendations"], list)
        
        # Basic reward for participation
        reward += 0.1
        
        # Reward for number of critique points (up to 0.4)
        reward += min(0.4, points_count * 0.1)
        
        # Reward for providing a score (up to 0.2)
        if has_score:
            reward += 0.2
            
        # Reward for providing recommendations (up to 0.3)
        if has_recommendations:
            reward += min(0.3, len(critique["recommendations"]) * 0.1)
        
        return reward * 0.3  # Stage 2 weight: 30%
    
    def _calculate_stage3_reward(self, consensus: Dict[str, Any], analyses: List[Dict[str, Any]]) -> float:
        """Calculate reward for a stage 3 consensus"""
        reward = 0.0
        
        # Check if the consensus has a final verdict
        has_verdict = "final_verdict" in consensus
        has_confidence = "confidence" in consensus and isinstance(consensus["confidence"], (int, float))
        has_reasoning = "reasoning" in consensus and len(consensus.get("reasoning", "")) > 100
        
        # Check if sources are cited
        cited_sources = consensus.get("cited_sources", [])
        sources_count = len(cited_sources) if isinstance(cited_sources, list) else 0
        
        # Basic reward for participation
        reward += 0.1
        
        # Reward for providing a verdict (up to 0.3)
        if has_verdict:
            reward += 0.3
            
        # Reward for providing confidence score (up to 0.2)
        if has_confidence:
            reward += 0.2
            
        # Reward for detailed reasoning (up to 0.3)
        if has_reasoning:
            reward += 0.3
            
        # Reward for citing sources (up to 0.1)
        reward += min(0.1, sources_count * 0.02)
        
        return reward * 0.3  # Stage 3 weight: 30%

def print_header():
    """Print colorful header for the demo"""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.YELLOW}🧠 SWARM RL ANTI-MISINFORMATION SYSTEM 🧠")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print("\nThis demo showcases a decentralized swarm of AI models that collaborate")
    print("to detect misinformation through three-stage reinforcement learning:\n")
    print(f"{Fore.GREEN}1. {Fore.WHITE}Analysis: {Fore.CYAN}Independent fact-checking by multiple peers")
    print(f"{Fore.GREEN}2. {Fore.WHITE}Critique: {Fore.CYAN}Cross-evaluation of analyses for flaws and insights")
    print(f"{Fore.GREEN}3. {Fore.WHITE}Consensus: {Fore.CYAN}Formation of a unified verdict with confidence score")
    print(f"\n{Fore.YELLOW}Each peer earns rewards based on the quality of their contributions.{Style.RESET_ALL}")

def print_stage_header(stage_num, stage_name):
    """Print stage header"""
    print(f"\n{Fore.CYAN}{'=' * 30}")
    print(f"{Fore.GREEN}STAGE {stage_num}: {Fore.YELLOW}{stage_name.upper()}")
    print(f"{Fore.CYAN}{'=' * 30}{Style.RESET_ALL}")

def print_peer_output(peer_name, model_type, output, reward=None):
    """Print peer output with formatting"""
    color = Fore.MAGENTA if model_type == "openai" else Fore.BLUE
    print(f"\n{color}[PEER: {peer_name} ({model_type.upper()})]")
    if reward is not None:
        print(f"{Fore.YELLOW}[REWARD: {reward:.2f}]{Style.RESET_ALL}")
    
    # Format the output nicely
    if isinstance(output, dict):
        # For each key in the output, print it nicely
        for key, value in output.items():
            if key == "reward":  # Skip as we already printed it
                continue
                
            print(f"{Fore.GREEN}{key}: {Style.RESET_ALL}", end="")
            
            if isinstance(value, list):
                print("")
                for i, item in enumerate(value):
                    print(f"  {Fore.CYAN}[{i+1}]{Style.RESET_ALL} {item}")
            else:
                print(f"{value}")
    else:
        print(output)
    print(f"{Style.RESET_ALL}")

def print_rewards_leaderboard(peers):
    """Print the rewards leaderboard"""
    print(f"\n{Fore.CYAN}{'=' * 50}")
    print(f"{Fore.YELLOW}🏆 REWARDS LEADERBOARD 🏆")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{'PEER':<15} {'ANALYSIS':<10} {'CRITIQUE':<10} {'CONSENSUS':<10} {'TOTAL':<10}{Style.RESET_ALL}")
    print(f"{'-' * 60}")
    
    # Sort peers by total reward
    sorted_peers = sorted(peers, key=lambda p: p.rewards["total"], reverse=True)
    
    for peer in sorted_peers:
        color = Fore.MAGENTA if peer.model_type == "openai" else Fore.BLUE
        print(f"{color}{peer.name:<15} {peer.rewards['stage1']:<10.2f} {peer.rewards['stage2']:<10.2f} {peer.rewards['stage3']:<10.2f} {peer.rewards['total']:<10.2f}{Style.RESET_ALL}")

def create_json_output(consensus, analyses, critiques, peers, claim=None):
    """Create a JSON string output suitable for blockchain storage"""
    verdict = consensus.get("final_verdict", "UNCERTAIN")
    confidence = consensus.get("confidence", 0.0)
    reasoning = consensus.get("reasoning", "")
    cited_sources = consensus.get("cited_sources", [])
    
    # Create the main verdict object
    result = {
        "claim": claim,  # Include the original claim
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "cited_sources": cited_sources,
        "consensus_peer": consensus.get("peer_name", "unknown"),
        "timestamp": time.time(),
        
        # Include all analyses in summarized form
        "analyses_summary": [
            {
                "peer": peers[i].name,
                "model": peers[i].model_type,
                "assessment": a.get("assessment", "unknown"),
                "confidence": a.get("confidence", 0.0),
                "reward": a.get("reward", 0.0)
            }
            for i, a in enumerate(analyses) if i < len(peers)
        ],
        
        # Include critiques in summarized form
        "critiques_summary": [
            {
                "peer": c.get("peer_name", "unknown"),
                "target_peer": c.get("target_peer", "unknown"),
                "score": c.get("score", 0.0),
                "reward": c.get("reward", 0.0)
            }
            for c in critiques
        ],
        
        # Add rewards leaderboard
        "rewards": [
            {
                "peer": peer.name,
                "model": peer.model_type,
                "analysis_reward": peer.rewards["stage1"],
                "critique_reward": peer.rewards["stage2"],
                "consensus_reward": peer.rewards["stage3"],
                "total_reward": peer.rewards["total"]
            }
            for peer in sorted(peers, key=lambda p: p.rewards["total"], reverse=True)
        ]
    }
    
    return json.dumps(result, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Demo for Swarm RL Misinformation Detection")
    parser.add_argument("--api-keys", type=str, default="api_keys.json", help="Path to API keys file")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG to enhance analysis")
    parser.add_argument("--use-google", action="store_true", help="Use Google Search API for context")
    parser.add_argument("--num-peers", type=int, default=4, help="Number of peers to simulate")
    parser.add_argument("--json-only", action="store_true", help="Output only the JSON result (for programmatic use)")
    parser.add_argument("--claim", type=str, help="Claim to analyze (if provided, bypasses interactive input)")
    args = parser.parse_args()
    
    try:
        # Load API keys
        api_keys = load_api_keys(args.api_keys)
        logger.info(f"Loaded API keys from {args.api_keys}")
        
        # Initialize context provider based on user choice
        context_provider = None
        
        # Initialize RAG if requested
        if args.use_rag:
            from anti_misinfo_swarm.models.rag_retriever import RAGRetriever
            from anti_misinfo_swarm.init_knowledge_base import KnowledgeBaseInitializer
            
            # Check if knowledge base exists, if not initialize it
            kb_dir = Path("knowledge_base")
            if not kb_dir.exists() or not (kb_dir / "index" / "faiss.index").exists():
                logger.info("Knowledge base not found. Initializing...")
                initializer = KnowledgeBaseInitializer()
                initializer.initialize()
            
            context_provider = RAGRetriever(kb_path="knowledge_base")
            logger.info("RAG enabled for enhanced analysis")
        
        # Initialize Google context provider if requested
        elif args.use_google:
            from anti_misinfo_swarm.models.google_context_provider import GoogleContextProvider
            
            # Check if Google API keys are valid
            if "google" not in api_keys or api_keys["google"]["api_key"] == "YOUR_GOOGLE_API_KEY":
                logger.error("Invalid Google API key. Please update your api_keys.json file.")
                print(f"{Fore.RED}ERROR: Google API keys not configured. Please update api_keys.json with valid Google API key and CSE ID.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}You can get these at: https://developers.google.com/custom-search/v1/introduction{Style.RESET_ALL}")
                return
                
            context_provider = GoogleContextProvider(
                api_key=api_keys["google"]["api_key"],
                cse_id=api_keys["google"]["cse_id"]
            )
            logger.info("Google Search API enabled for context retrieval")
        
        # Create simulated peers
        peers = []
        for i in range(args.num_peers):
            model_type = "openai" if i % 2 == 0 else "perplexity"
            api_key = api_keys["openai"] if model_type == "openai" else api_keys["perplexity"]
            
            # Generate animal-based peer names
            animals = ["Lion", "Tiger", "Eagle", "Dolphin", "Wolf", "Bear", "Hawk", "Falcon"]
            name = f"{random.choice(animals)}-{random.randint(1000, 9999)}"
            
            peer = SwarmPeer(name, model_type, api_key, context_provider)
            peers.append(peer)
        
        # Print demo header
        if not args.json_only:
            print_header()
        
        # If a claim was provided via command line
        if args.claim:
            claim = args.claim
            if not args.json_only:
                print(f"\n{Fore.YELLOW}Analyzing claim: {Fore.WHITE}{claim}{Style.RESET_ALL}")
            
            # Process the claim
            process_claim(claim, peers, context_provider, args.json_only)
        else:
            # Main demo loop for interactive use
            while True:
                # Get user input
                claim = input(f"\n{Fore.GREEN}Enter a claim to analyze (or 'exit' to quit): {Style.RESET_ALL}")
                if claim.lower() in ['exit', 'quit', 'q']:
                    break
                    
                if not claim.strip():
                    continue
                
                if not args.json_only:
                    print(f"\n{Fore.YELLOW}Analyzing claim: {Fore.WHITE}{claim}{Style.RESET_ALL}")
                
                # Process the claim
                process_claim(claim, peers, context_provider, args.json_only)
    
    except KeyboardInterrupt:
        if not args.json_only:
            print(f"\n{Fore.YELLOW}Demo terminated by user.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        if not args.json_only:
            import traceback
            traceback.print_exc()
        else:
            print(json.dumps({"error": str(e)}))
    finally:
        # Final leaderboard
        if 'peers' in locals() and not args.json_only:
            print_rewards_leaderboard(peers)

def process_claim(claim, peers, context_provider, json_only=False):
    # Get context if available
    context = []
    if context_provider:
        context = context_provider.get_context_passages(claim)
        if context and not json_only:
            context_type = "knowledge base" if isinstance(context_provider, RAGRetriever) else "Google Search"
            print(f"\n{Fore.CYAN}Found {len(context)} relevant facts from {context_type}:{Style.RESET_ALL}")
            for i, fact in enumerate(context):
                print(f"{Fore.GREEN}[{i+1}] {Fore.WHITE}{fact[:100]}...{Style.RESET_ALL}")
    
    # STAGE 1: Analysis
    if not json_only:
        print_stage_header(1, "Independent Analysis")
    
    # Each peer analyzes the claim
    analyses = []
    for peer in peers:
        # Simulate peer thinking time
        if not json_only:
            print(f"\n{Fore.CYAN}Peer {peer.name} ({peer.model_type}) analyzing...{Style.RESET_ALL}")
            time.sleep(0.5 + random.random() * 1.5)
        
        # Perform analysis
        analysis = peer.analyze_claim(claim, context)
        analyses.append(analysis)
        
        # Print the analysis with the reward
        if not json_only:
            print_peer_output(peer.name, peer.model_type, analysis, analysis.get("reward", 0.0))
    
    # STAGE 2: Critique
    if not json_only:
        print_stage_header(2, "Cross-Critique")
    
    # Each peer critiques other peers' analyses
    critiques = []
    for i, peer in enumerate(peers):
        # Choose a random analysis from another peer to critique
        other_indices = [j for j in range(len(peers)) if j != i]
        if not other_indices:  # If there's only one peer
            continue
            
        other_idx = random.choice(other_indices)
        other_peer = peers[other_idx]
        analysis_to_critique = analyses[other_idx]
        
        # Simulate peer thinking time
        if not json_only:
            print(f"\n{Fore.CYAN}Peer {peer.name} critiquing {other_peer.name}'s analysis...{Style.RESET_ALL}")
            time.sleep(0.5 + random.random() * 1.0)
        
        # Perform critique
        critique = peer.critique_analysis(analysis_to_critique, claim)
        critique["target_peer"] = other_peer.name
        critiques.append(critique)
        
        # Print the critique with the reward
        if not json_only:
            print_peer_output(peer.name, peer.model_type, critique, critique.get("reward", 0.0))
    
    # STAGE 3: Consensus
    if not json_only:
        print_stage_header(3, "Consensus Formation")
    
    # Select a random peer to form the consensus
    consensus_peer = random.choice(peers)
    
    # Simulate peer thinking time
    if not json_only:
        print(f"\n{Fore.CYAN}Peer {consensus_peer.name} forming consensus...{Style.RESET_ALL}")
        time.sleep(1.0 + random.random() * 2.0)
    
    # Form consensus
    consensus = consensus_peer.form_consensus(analyses, critiques, claim)
    
    # Print the consensus with the reward
    if not json_only:
        print_peer_output(consensus_peer.name, consensus_peer.model_type, consensus, consensus.get("reward", 0.0))
    
    # Final verdict summary
    verdict = consensus.get("final_verdict", "UNCERTAIN")
    confidence = consensus.get("confidence", 0.0)
    
    # Generate JSON output for blockchain
    json_output = create_json_output(consensus, analyses, critiques, peers, claim)
    
    if json_only:
        # If json-only mode, print just the JSON output without formatting
        print(json_output)
    else:
        # Print formatted output for interactive mode
        print(f"\n{Fore.CYAN}{'=' * 50}")
        print(f"{Fore.YELLOW}📊 SWARM VERDICT:")
        
        if verdict == "MISINFORMATION":
            print(f"{Fore.RED}❌ This claim is likely MISINFORMATION (Confidence: {confidence:.2f})")
        elif verdict == "ACCURATE":
            print(f"{Fore.GREEN}✅ This claim is likely ACCURATE (Confidence: {confidence:.2f})")
        else:
            print(f"{Fore.YELLOW}⚠️ This claim's accuracy is UNCERTAIN (Confidence: {confidence:.2f})")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'=' * 50}")
        print(f"{Fore.YELLOW}🔗 JSON OUTPUT FOR BLOCKCHAIN:{Style.RESET_ALL}")
        print(f"{json_output}")
        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
        
        # Print rewards leaderboard
        print_rewards_leaderboard(peers)
    
    return json_output

# Add a helper function to run from external code
def get_swarm_verdict(claim, num_peers=4, use_rag=False, api_keys_path="api_keys.json"):
    """
    Get swarm verdict as JSON for a given claim - suitable for external calls
    
    Args:
        claim: The claim to analyze
        num_peers: Number of peers to use
        use_rag: Whether to use RAG for analysis
        api_keys_path: Path to API keys file
        
    Returns:
        JSON string with the verdict and details
    """
    import sys
    import subprocess
    
    # Construct command to run the script in json-only mode
    cmd = [
        sys.executable, 
        __file__, 
        "--json-only",
        "--num-peers", str(num_peers),
        "--api-keys", api_keys_path
    ]
    
    if use_rag:
        cmd.append("--use-rag")
    
    # Run the process and capture output
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Send the claim to the process
    stdout, stderr = process.communicate(input=claim)
    
    if process.returncode != 0:
        return json.dumps({
            "error": f"Process failed with exit code {process.returncode}",
            "stderr": stderr
        })
    
    return stdout.strip()

if __name__ == "__main__":
    main() 