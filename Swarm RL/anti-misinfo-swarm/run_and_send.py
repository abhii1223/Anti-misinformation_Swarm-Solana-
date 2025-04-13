#!/usr/bin/env python3
"""
Run Swarm RL demo with full output and then send results to Solana.
This preserves the complete demo experience while also storing results on-chain.
"""

import os
import sys
import json
import argparse
import subprocess
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

def run_swarm_demo(claim, num_peers=4, use_rag=False, use_google=False, api_keys_path=None):
    """Run the swarm demo with the given parameters"""
    cmd = [sys.executable, str(Path(__file__).parent / "swarm_demo.py")]
    
    if num_peers:
        cmd.extend(["--num-peers", str(num_peers)])
    
    if use_rag:
        cmd.append("--use-rag")
    
    if use_google:
        cmd.append("--use-google")
    
    if api_keys_path:
        cmd.extend(["--api-keys", str(api_keys_path)])
    
    # Run in json-only mode to capture the result
    cmd.append("--json-only")
    
    # Set up environment with API keys directly if available
    env = os.environ.copy()
    if api_keys_path:
        keys = load_api_keys(api_keys_path)
        if keys:
            # Set API keys as environment variables
            if "openai" in keys and keys["openai"] != "YOUR_OPENAI_API_KEY":
                env["OPENAI_API_KEY"] = keys["openai"]
            if "perplexity" in keys and keys["perplexity"] != "YOUR_PERPLEXITY_API_KEY":
                env["PERPLEXITY_API_KEY"] = keys["perplexity"]
    
    try:
        # A different approach: Let's echo the claim and pipe it to the process
        # This avoids EOF issues that can occur with communicate
        echo_cmd = ["echo", claim]
        demo_cmd = cmd
        
        echo_process = subprocess.Popen(
            echo_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        demo_process = subprocess.Popen(
            demo_cmd,
            stdin=echo_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        # Close the pipe in the echo process
        echo_process.stdout.close()
        
        # Get the output from the demo process
        stdout, stderr = demo_process.communicate()
        
        # Check for errors
        if demo_process.returncode != 0:
            print(f"Error running swarm_demo.py: {stderr}")
            print("Trying fallback method...")
            return run_swarm_demo_fallback(claim, num_peers, use_rag, use_google, api_keys_path, env)
        
        # Return the JSON output
        try:
            # Try to parse as JSON to validate
            json_data = json.loads(stdout.strip())
            return stdout.strip()
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON response from swarm_demo.py")
            print(f"Output: {stdout}")
            print(f"Error: {stderr}")
            print("Trying fallback method...")
            return run_swarm_demo_fallback(claim, num_peers, use_rag, use_google, api_keys_path, env)
    except Exception as e:
        print(f"Error executing the command: {e}")
        print("Trying fallback method...")
        return run_swarm_demo_fallback(claim, num_peers, use_rag, use_google, api_keys_path, env)

def run_swarm_demo_fallback(claim, num_peers=4, use_rag=False, use_google=False, api_keys_path=None, env=None):
    """Fallback method to run the swarm demo using a temporary file for input"""
    import tempfile
    
    cmd = [sys.executable, str(Path(__file__).parent / "swarm_demo.py")]
    
    if num_peers:
        cmd.extend(["--num-peers", str(num_peers)])
    
    if use_rag:
        cmd.append("--use-rag")
    
    if use_google:
        cmd.append("--use-google")
    
    if api_keys_path:
        cmd.extend(["--api-keys", str(api_keys_path)])
    
    # Run in json-only mode to capture the result
    cmd.append("--json-only")
    
    # If no environment provided, create one
    if env is None:
        env = os.environ.copy()
        if api_keys_path:
            keys = load_api_keys(api_keys_path)
            if keys:
                if "openai" in keys and keys["openai"] != "YOUR_OPENAI_API_KEY":
                    env["OPENAI_API_KEY"] = keys["openai"]
                if "perplexity" in keys and keys["perplexity"] != "YOUR_PERPLEXITY_API_KEY":
                    env["PERPLEXITY_API_KEY"] = keys["perplexity"]
    
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env
            )
            
            # Get output
            stdout, stderr = process.communicate()
            
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            # Check for errors
            if process.returncode != 0:
                print(f"Fallback method also failed: {stderr}")
                return None
            
            # Try to parse JSON
            try:
                json_data = json.loads(stdout.strip())
                return stdout.strip()
            except json.JSONDecodeError:
                print(f"Fallback method produced invalid JSON: {stdout}")
                return None
    except Exception as e:
        print(f"Error in fallback method: {e}")
        return None

def run_interactive_demo_fallback(claim, num_peers=4, use_rag=False, use_google=False, api_keys_path=None, env=None):
    """Fallback method to run the interactive demo using a temporary file"""
    import tempfile
    
    # Create the demo command
    demo_cmd = [sys.executable, str(Path(__file__).parent / "swarm_demo.py")]
    
    if num_peers:
        demo_cmd.extend(["--num-peers", str(num_peers)])
    
    if use_rag:
        demo_cmd.append("--use-rag")
    
    if use_google:
        demo_cmd.append("--use-google")
    
    if api_keys_path:
        demo_cmd.extend(["--api-keys", str(api_keys_path)])
    
    # If no environment provided, create one
    if env is None:
        env = os.environ.copy()
        if api_keys_path:
            keys = load_api_keys(api_keys_path)
            if keys:
                if "openai" in keys and keys["openai"] != "YOUR_OPENAI_API_KEY":
                    env["OPENAI_API_KEY"] = keys["openai"]
                if "perplexity" in keys and keys["perplexity"] != "YOUR_PERPLEXITY_API_KEY":
                    env["PERPLEXITY_API_KEY"] = keys["perplexity"]
    
    try:
        # Create a temporary file with the claim
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(claim)
            temp_file_path = temp_file.name
        
        # Use the temporary file as input
        with open(temp_file_path, 'r') as input_file:
            # Run the demo process
            process = subprocess.Popen(
                demo_cmd,
                stdin=input_file,
                stdout=None,  # Display to console
                stderr=None,  # Display to console
                universal_newlines=True,
                env=env
            )
            
            # Wait for completion
            process.wait()
            
            # Clean up
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            if process.returncode != 0:
                print(f"Warning: Demo process exited with code {process.returncode}")
                return False
            
            return True
    except Exception as e:
        print(f"Error in fallback method: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Swarm RL demo and send results to Solana")
    parser.add_argument("--claim", type=str, help="Claim to fact-check")
    parser.add_argument("--num-peers", type=int, default=4, help="Number of peers to use")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG to enhance analysis")
    parser.add_argument("--use-google", action="store_true", help="Use Google Search API for context")
    parser.add_argument("--sender", type=str, help="Path to sender's keypair file")
    parser.add_argument("--receiver", type=str, help="Receiver's address (if different from sender)")
    parser.add_argument("--lamports", type=int, default=5000, help="Amount to send in lamports (default: 5000)")
    parser.add_argument("--no-send", action="store_true", help="Don't send to Solana, just run the demo")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of running the actual demo")
    parser.add_argument("--api-keys", type=str, default=str(Path(__file__).parent / "api_keys.json"), 
                       help="Path to API keys file")
    args = parser.parse_args()
    
    # Get claim from input if not provided as argument
    claim = args.claim
    if not claim:
        claim = input("Enter claim to fact-check: ")
    
    # Set API keys path
    api_keys_path = args.api_keys
    
    # Load API keys directly
    if not args.mock:
        keys = load_api_keys(api_keys_path)
        if not keys:
            print("Error loading API keys.")
            use_mock = input("Run with mock data instead? (y/n): ")
            if use_mock.lower() == 'y':
                args.mock = True
            else:
                print("Please check your API keys file and try again.")
                return
    
    if not args.mock:
        # First, run the regular swarm_demo for visual output
        print(f"Running Swarm RL demo for claim: '{claim}'")
        print("=" * 80)
        
        demo_cmd = [sys.executable, str(Path(__file__).parent / "swarm_demo.py")]
        
        if args.num_peers:
            demo_cmd.extend(["--num-peers", str(args.num_peers)])
        
        if args.use_rag:
            demo_cmd.append("--use-rag")
        
        if args.use_google:
            demo_cmd.append("--use-google")
        
        # Add API keys path
        demo_cmd.extend(["--api-keys", api_keys_path])
        
        # Set up environment with API keys directly
        env = os.environ.copy()
        if keys:
            # Set API keys as environment variables
            if "openai" in keys and keys["openai"] != "YOUR_OPENAI_API_KEY":
                env["OPENAI_API_KEY"] = keys["openai"]
            if "perplexity" in keys and keys["perplexity"] != "YOUR_PERPLEXITY_API_KEY":
                env["PERPLEXITY_API_KEY"] = keys["perplexity"]
        
        try:
            # Using the pipe approach to avoid EOF errors
            echo_cmd = ["echo", claim]
            
            # Start the echo process to generate input
            echo_process = subprocess.Popen(
                echo_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Run the regular demo with full output
            demo_process = subprocess.Popen(
                demo_cmd,
                stdin=echo_process.stdout,
                stdout=None,  # Display to console
                stderr=None,  # Display to console
                universal_newlines=True,
                env=env
            )
            
            # Close the pipe in the echo process
            echo_process.stdout.close()
            
            # Wait for completion
            demo_process.wait()
            
            if demo_process.returncode != 0:
                print(f"Warning: Demo process exited with code {demo_process.returncode}")
                print("Trying fallback method...")
                run_interactive_demo_fallback(claim, args.num_peers, args.use_rag, args.use_google, api_keys_path, env)
        except Exception as e:
            print(f"Error during demo execution: {e}")
            print("Trying fallback method...")
            run_interactive_demo_fallback(claim, args.num_peers, args.use_rag, args.use_google, api_keys_path, env)
            print("Continuing with JSON generation...")
    else:
        print(f"Using mock data for claim: '{claim}'")
    
    if args.no_send:
        print("\nSkipping Solana transaction as requested.")
        return
    
    # Get verdict JSON (from swarm demo or generate mock)
    print("\n" + "=" * 80)
    print("Getting verdict for Solana transaction...")
    
    if args.mock:
        verdict_data = generate_mock_verdict(claim)
        verdict_json = json.dumps(verdict_data)
    else:
        verdict_json = run_swarm_demo(claim, args.num_peers, args.use_rag, args.use_google, api_keys_path)
    
    if not verdict_json:
        print("Failed to get verdict JSON. Cannot send to Solana.")
        return
    
    # Flatten the JSON
    flattened_json = flatten_json(verdict_json)
    print("\n--- Flattened JSON for Solana Memo ---")
    print(flattened_json)
    
    # Confirm before sending to blockchain
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
    
    print(f"\nTransaction result: {result}")
    print(f"\nVerdict has been stored on Solana blockchain.")

if __name__ == "__main__":
    main()

 