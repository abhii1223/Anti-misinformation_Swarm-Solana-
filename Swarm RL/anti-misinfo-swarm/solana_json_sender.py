#!/usr/bin/env python3
"""
Solana JSON sender for Swarm RL fact checking system.
This script allows sending larger JSON data to Solana by splitting it
into multiple transactions if needed.
"""

import json
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

# Import the swarm demo functionality
from swarm_demo import get_swarm_verdict

# Maximum memo size (in bytes)
MAX_MEMO_SIZE = 500  # Conservative limit

def chunk_json(json_data, max_size=MAX_MEMO_SIZE):
    """
    Split JSON into chunks that fit within the memo size limit
    
    Args:
        json_data: JSON string to split
        max_size: Maximum size of each chunk in bytes
        
    Returns:
        List of chunks with metadata
    """
    if isinstance(json_data, dict):
        data_str = json.dumps(json_data)
    else:
        data_str = json_data
    
    # Calculate number of chunks needed
    total_size = len(data_str.encode('utf-8'))
    num_chunks = (total_size + max_size - 1) // max_size
    
    chunks = []
    for i in range(num_chunks):
        start = i * max_size
        end = min((i + 1) * max_size, total_size)
        
        # Create chunk with metadata
        chunk = {
            "type": "swarm_verdict",
            "chunk": i + 1,
            "total_chunks": num_chunks,
            "timestamp": int(datetime.now().timestamp()),
            "data": data_str[start:end]
        }
        
        chunks.append(json.dumps(chunk, separators=(',', ':')))
    
    return chunks

def send_to_solana(memo, from_keypair=None, to_address=None, lamports=5000):
    """
    Send a transaction to Solana with the memo attached
    
    Args:
        memo: The memo string to attach to the transaction
        from_keypair: Path to sender's keypair file (defaults to config)
        to_address: Recipient address (defaults to sender if none provided)
        lamports: Amount to send in lamports (default minimal amount)
        
    Returns:
        Transaction signature or error message
    """
    try:
        cmd = ["solana", "transfer"]
        
        # Add keypair if provided
        if from_keypair:
            cmd.extend(["-k", from_keypair])
        
        # Set recipient address
        if not to_address:
            # If no recipient specified, get the current configured address
            result = subprocess.run(["solana", "address"], 
                                   capture_output=True, 
                                   text=True, 
                                   check=True)
            to_address = result.stdout.strip()
        
        # Add the recipient, amount, and memo
        cmd.extend([
            to_address,
            str(lamports / 1000000),  # Convert lamports to SOL
            "--allow-unfunded-recipient",
            "--with-memo", memo
        ])
        
        # Execute the command
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Extract signature from output
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if "Signature:" in line:
                return line.replace("Signature:", "").strip()
        
        return result.stdout.strip()
        
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Send full Swarm RL fact-checking JSON to Solana blockchain")
    parser.add_argument("--claim", type=str, help="Claim to fact-check")
    parser.add_argument("--num-peers", type=int, default=4, help="Number of peers to use")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for analysis")
    parser.add_argument("--sender", type=str, help="Path to sender's keypair file")
    parser.add_argument("--receiver", type=str, help="Receiver's address (if different from sender)")
    parser.add_argument("--lamports", type=int, default=5000, help="Amount to send in lamports (default: 5000)")
    parser.add_argument("--include-full", action="store_true", help="Include the full detailed JSON (may require multiple transactions)")
    args = parser.parse_args()
    
    # Get claim from input if not provided as argument
    claim = args.claim
    if not claim:
        claim = input("Enter claim to fact-check: ")
    
    print(f"Analyzing claim: {claim}")
    print("Running Swarm RL fact-checking...")
    
    # Get the verdict
    verdict_json = get_swarm_verdict(claim, args.num_peers, args.use_rag)
    verdict_data = json.loads(verdict_json)
    
    # Create summary version for the first transaction
    summary = {
        "claim": claim[:100] + "..." if len(claim) > 100 else claim,
        "verdict": verdict_data.get("verdict", "UNCERTAIN"),
        "confidence": verdict_data.get("confidence", 0.0),
        "timestamp": verdict_data.get("timestamp", int(datetime.now().timestamp())),
        "consensus_peer": verdict_data.get("consensus_peer", "unknown"),
        "total_peers": len(verdict_data.get("analyses_summary", [])),
        "avg_reward": sum(peer.get("total_reward", 0.0) for peer in verdict_data.get("rewards", [])) / 
                     max(1, len(verdict_data.get("rewards", [])))
    }
    
    # Convert to compact JSON
    summary_json = json.dumps(
        {"type": "swarm_verdict_summary", "data": summary},
        separators=(',', ':')
    )
    
    print("\n--- Summary JSON for first transaction ---")
    print(summary_json)
    
    # Confirm before sending to blockchain
    confirm = input("\nSend this verdict to Solana blockchain? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Send summary transaction
    print("\nSending summary transaction to Solana...")
    result = send_to_solana(
        summary_json,
        from_keypair=args.sender,
        to_address=args.receiver,
        lamports=args.lamports
    )
    
    print(f"\nSummary transaction result: {result}")
    
    # If requested, send full details in additional transactions
    if args.include_full:
        print("\nPreparing to send full details...")
        
        # Split the full JSON into chunks
        chunks = chunk_json(verdict_json)
        
        print(f"JSON will be sent in {len(chunks)} transactions")
        
        # Send each chunk
        signatures = []
        for i, chunk in enumerate(chunks):
            print(f"\nSending chunk {i+1} of {len(chunks)}...")
            
            sig = send_to_solana(
                chunk,
                from_keypair=args.sender,
                to_address=args.receiver,
                lamports=args.lamports
            )
            signatures.append(sig)
            
            print(f"Chunk {i+1} result: {sig}")
            
            # Brief pause between transactions
            if i < len(chunks) - 1:
                time.sleep(1)
        
        print("\nAll transactions completed:")
        for i, sig in enumerate(signatures):
            print(f"Chunk {i+1}: {sig}")

if __name__ == "__main__":
    main() 