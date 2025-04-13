#!/usr/bin/env python3
"""
One-shot Swarm RL demo - run once and send to Solana.
Simplified script that runs just one claim through the demo and sends it to Solana.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import solana sender functionality
from solana_sender import flatten_json, send_to_solana

def load_api_keys(api_keys_path):
    """Load API keys from file path"""
    try:
        with open(api_keys_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading API keys: {e}")
        return None

def generate_mock_verdict(claim):
    """Generate a mock verdict for testing without API keys"""
    return {
        "verdict": "UNCERTAIN",
        "confidence": 0.65,
        "reasoning": "This is a mock verdict generated for testing without API keys.",
        "cited_sources": ["Mock source 1", "Mock source 2"],
        "consensus_peer": "MockPeer-1234",
        "timestamp": int(datetime.now().timestamp()),
        "claim": claim,
        "analyses_summary": [
            {
                "peer": "MockPeer-1",
                "model": "openai",
                "assessment": "UNCERTAIN",
                "confidence": 0.6,
                "reward": 0.4
            },
            {
                "peer": "MockPeer-2",
                "model": "perplexity",
                "assessment": "UNCERTAIN",
                "confidence": 0.7,
                "reward": 0.5
            }
        ],
        "rewards": [
            {
                "peer": "MockPeer-1",
                "model": "openai",
                "total_reward": 0.8
            },
            {
                "peer": "MockPeer-2", 
                "model": "perplexity",
                "total_reward": 0.9
            }
        ]
    }

def run_demo_with_temp_file(claim, cmd, env=None):
    """Run demo using a temporary file to avoid EOF issues"""
    try:
        # Create a temporary file with the claim
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(claim)
            temp_file_path = temp_file.name
        
        # Use the temporary file as input
        with open(temp_file_path, 'r') as input_file:
            # Run the demo process
            process = subprocess.Popen(
                cmd,
                stdin=input_file,
                stdout=subprocess.PIPE if "--json-only" in cmd else None,
                stderr=subprocess.PIPE if "--json-only" in cmd else None,
                universal_newlines=True,
                env=env
            )
            
            # Get output if needed
            stdout, stderr = process.communicate() if "--json-only" in cmd else (None, None)
            
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            # Return result
            if "--json-only" in cmd:
                if process.returncode != 0:
                    print(f"Error running demo: {stderr}")
                    return None
                return stdout.strip()
            else:
                return process.returncode == 0
    except Exception as e:
        print(f"Error running demo: {e}")
        return None if "--json-only" in cmd else False

def run_demo_direct(claim, cmd, env=None, with_json=False):
    """Run the demo by passing the claim directly as an argument and environment variable"""
    try:
        # Set up environment
        if env is None:
            env = os.environ.copy()
        
        # Add the claim as an environment variable
        env["SWARM_CLAIM"] = claim
        
        # Add claim parameter to command - better than stdin
        cmd = cmd + ["--claim", claim]
        
        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if with_json else None,
            stderr=subprocess.PIPE if with_json else None,
            universal_newlines=True,
            env=env
        )
        
        # Get output if needed
        stdout, stderr = process.communicate()
        
        # Return result
        if with_json:
            if process.returncode != 0:
                print(f"Error running demo: {stderr}")
                return None
            return stdout.strip()
        else:
            return process.returncode == 0
    except Exception as e:
        print(f"Error running demo: {e}")
        return None if with_json else False

def main():
    parser = argparse.ArgumentParser(description="One-shot Swarm RL demo - run once and send to Solana")
    parser.add_argument("claim", type=str, help="Claim to fact-check")
    parser.add_argument("--num-peers", type=int, default=4, help="Number of peers to use")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG to enhance analysis")
    parser.add_argument("--use-google", action="store_true", help="Use Google Search API for context")
    parser.add_argument("--sender", type=str, help="Path to sender's keypair file")
    parser.add_argument("--receiver", type=str, help="Receiver's address (if different from sender)")
    parser.add_argument("--lamports", type=int, default=5000, help="Amount to send in lamports (default: 5000)")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of running the actual demo")
    parser.add_argument("--api-keys", type=str, default=str(Path(__file__).parent / "api_keys.json"), 
                       help="Path to API keys file")
    args = parser.parse_args()
    
    claim = args.claim.strip()
    if not claim:
        print("Error: Claim cannot be empty")
        return
    
    # Set API keys path
    api_keys_path = args.api_keys
    
    # Load API keys directly
    if not args.mock:
        keys = load_api_keys(api_keys_path)
        if not keys:
            print("Error loading API keys.")
            print("Using mock data instead.")
            args.mock = True
    
    # Set up environment with API keys
    env = os.environ.copy()
    if not args.mock and keys:
        if "openai" in keys and keys["openai"] != "YOUR_OPENAI_API_KEY":
            env["OPENAI_API_KEY"] = keys["openai"]
        if "perplexity" in keys and keys["perplexity"] != "YOUR_PERPLEXITY_API_KEY":
            env["PERPLEXITY_API_KEY"] = keys["perplexity"]
    
    # Run the swarm demo for visual output
    if not args.mock:
        print(f"Running Swarm RL demo for claim: '{claim}'")
        print("=" * 80)
        
        # Create demo command
        demo_cmd = [sys.executable, str(Path(__file__).parent / "swarm_demo.py")]
        
        if args.num_peers:
            demo_cmd.extend(["--num-peers", str(args.num_peers)])
        
        if args.use_rag:
            demo_cmd.append("--use-rag")
        
        if args.use_google:
            demo_cmd.append("--use-google")
        
        # Add API keys path
        demo_cmd.extend(["--api-keys", api_keys_path])
        
        # Run the demo with the claim - try direct approach first
        success = run_demo_direct(claim, demo_cmd, env)
        if not success:
            print("Direct approach failed. Trying with temp file...")
            success = run_demo_with_temp_file(claim, demo_cmd, env)
            if not success:
                print("Warning: Demo encountered issues. Continuing with JSON generation...")
    else:
        print(f"Using mock data for claim: '{claim}'")
    
    # Get verdict JSON (from swarm demo or generate mock)
    print("\n" + "=" * 80)
    print("Getting verdict for Solana transaction...")
    
    verdict_json = None
    if args.mock:
        verdict_data = generate_mock_verdict(claim)
        verdict_json = json.dumps(verdict_data)
    else:
        # Create JSON-only command
        json_cmd = [sys.executable, str(Path(__file__).parent / "swarm_demo.py")]
        
        if args.num_peers:
            json_cmd.extend(["--num-peers", str(args.num_peers)])
        
        if args.use_rag:
            json_cmd.append("--use-rag")
        
        if args.use_google:
            json_cmd.append("--use-google")
        
        json_cmd.extend(["--api-keys", api_keys_path, "--json-only"])
        
        # Run the demo in JSON-only mode - try direct approach first
        verdict_json = run_demo_direct(claim, json_cmd, env, with_json=True)
        if not verdict_json:
            print("Direct approach failed. Trying with temp file...")
            verdict_json = run_demo_with_temp_file(claim, json_cmd, env)
    
    if not verdict_json:
        print("Failed to get verdict JSON. Cannot send to Solana.")
        return
    
    # Flatten the JSON
    flattened_json = flatten_json(verdict_json)
    print("\n--- Flattened JSON for Solana Memo ---")
    print(flattened_json)
    
    # Send to Solana without further confirmation
    print("\nSending transaction to Solana...")
    result = send_to_solana(
        flattened_json,
        from_keypair=args.sender,
        to_address=args.receiver,
        lamports=args.lamports
    )
    
    print(f"\nTransaction result: {result}")
    print(f"\nVerdict has been stored on Solana blockchain.")

if __name__ == "__main__":
    main() 