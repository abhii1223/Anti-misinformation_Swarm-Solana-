#!/usr/bin/env python3
"""
Solana transaction sender for Swarm RL fact checking system.
This script takes the verdict JSON from swarm_demo.py, flattens it,
and sends it as a memo on a Solana transaction.
"""

import json
import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent))

# Import the swarm demo functionality
from swarm_demo import get_swarm_verdict

def flatten_json(json_data):
    """
    Flatten the JSON verdict into a format suitable for a Solana memo
    
    Args:
        json_data: Dictionary or JSON string to flatten
        
    Returns:
        Flattened string representation
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Get the claim from the data
    claim = data.get("claim", "")
    
    # Extract only the essential information to fit in a memo
    flattened = {
        "clm": claim[:50] + "..." if len(claim) > 50 else claim,
        "vrd": data.get("verdict", "UNCERTAIN"),
        "cnf": round(data.get("confidence", 0.0), 2),
        "ts": data.get("timestamp", int(datetime.now().timestamp())),
        "cp": data.get("consensus_peer", "")[:8],
        "tp": len(data.get("analyses_summary", [])),
        "rew": round(sum(peer.get("total_reward", 0.0) for peer in data.get("rewards", [])) / 
                   max(1, len(data.get("rewards", []))), 2)
    }
    
    # Convert to compact JSON string
    return json.dumps(flattened, separators=(',', ':'))

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
        signature = None
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if "Signature:" in line:
                signature = line.replace("Signature:", "").strip()
                break
        
        if not signature:
            signature = result.stdout.strip()
        
        # Return both the signature and additional useful information
        return {
            "signature": signature,
            "explorer_url": f"https://explorer.solana.com/tx/{signature}?cluster=devnet",
            "success": True,
            "from": from_keypair,
            "to": to_address,
            "amount": lamports / 1000000,
            "memo": memo
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "error": f"Error: {e.stderr.strip()}",
            "success": False
        }
    except Exception as e:
        return {
            "error": f"Error: {str(e)}",
            "success": False
        }

def main():
    parser = argparse.ArgumentParser(description="Send Swarm RL fact-checking verdict to Solana blockchain")
    parser.add_argument("--claim", type=str, help="Claim to fact-check")
    parser.add_argument("--json-file", type=str, help="Path to JSON file containing the verdict")
    parser.add_argument("--num-peers", type=int, default=4, help="Number of peers to use")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for analysis")
    parser.add_argument("--sender", type=str, help="Path to sender's keypair file")
    parser.add_argument("--receiver", type=str, help="Receiver's address (if different from sender)")
    parser.add_argument("--lamports", type=int, default=5000, help="Amount to send in lamports (default: 5000)")
    parser.add_argument("--auto-confirm", action="store_true", help="Skip confirmation prompt and send automatically")
    args = parser.parse_args()
    
    verdict_json = None
    
    # If JSON file is provided, load it
    if args.json_file:
        try:
            with open(args.json_file, 'r') as f:
                verdict_json = f.read().strip()
            print(f"Loaded verdict from {args.json_file}")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return
    # Otherwise, get claim from input if not provided as argument
    else:
        claim = args.claim
        if not claim:
            claim = input("Enter claim to fact-check: ")
        
        print(f"Analyzing claim: {claim}")
        print("Running Swarm RL fact-checking...")
        
        # Get the verdict
        verdict_json = get_swarm_verdict(claim, args.num_peers, args.use_rag)
    
    if not verdict_json:
        print("Error: No verdict data available.")
        return
    
    # Print full verdict for reference
    print("\n--- Full Verdict JSON ---")
    print(verdict_json)
    
    # Flatten the JSON
    flattened_json = flatten_json(verdict_json)
    print("\n--- Flattened JSON for Solana Memo ---")
    print(flattened_json)
    
    # Confirm before sending to blockchain (unless auto-confirm is enabled)
    if not args.auto_confirm:
        confirm = input("\nSend this verdict to Solana blockchain? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Send to Solana
    print("\nSending transaction to Solana...")
    result = send_to_solana(
        flattened_json,
        from_keypair=args.sender,
        to_address=args.receiver,
        lamports=args.lamports
    )
    
    if result["success"]:
        print(f"\n✅ Transaction successful!")
        print(f"Signature: {result['signature']}")
        print(f"View on Solana Explorer: {result['explorer_url']}")
        print(f"From: {result['from'] or 'default keypair'}")
        print(f"To: {result['to']}")
        print(f"Amount: {result['amount']} SOL")
        print(f"Memo contains verdict for: {json.loads(flattened_json)['clm']}")
    else:
        print(f"\n❌ Transaction failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 