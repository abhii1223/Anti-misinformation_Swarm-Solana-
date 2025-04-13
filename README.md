# Anti-Misinformation Swarm RL on Solana

This repository contains a decentralized fact-checking system that uses swarm intelligence and reinforcement learning to detect and combat misinformation. The results are stored on the Solana blockchain for transparency and immutability.

## Overview

The system simulates a swarm of AI agents (peers) that work together to analyze claims, critique each other's analyses, and form a consensus on the veracity of information. The process employs a three-stage approach:

1. **Independent Analysis**: Each peer independently analyzes the claim
2. **Cross-Critique**: Peers critique each other's analyses to identify gaps or errors
3. **Consensus Formation**: A final consensus is formed based on the analyses and critiques

The final verdict and all intermediate steps are then sent to the Solana blockchain for permanent, transparent record-keeping.

## Features

- Multi-agent AI system with simulated peers
- Integration with multiple LLM providers (OpenAI, Perplexity)
- Retrieval-augmented generation (RAG) capability for enhanced fact-checking
- Google Search API integration for real-time context
- Solana blockchain integration for immutable record-keeping
- Reinforcement learning component to improve peer performance over time

## Getting Started

### Prerequisites

- Python 3.8+
- Solana CLI tools (for blockchain integration)
- API keys for OpenAI, Perplexity (optional), and Google (optional)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/abhii1223/Anti-misinformation_Swarm-Solana-.git
   cd Anti-misinformation_Swarm-Solana-
   ```

2. Install the required packages:
   ```
   cd Swarm\ RL/anti-misinfo-swarm/
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Create an `api_keys.json` file based on the `api_keys.example.json` template
   - Add your API keys for the services you want to use

### Usage

#### Running the Demo

For the basic swarm demo:
```
python swarm_demo.py --claim "Your claim to fact-check"
```

Additional options:
- `--num-peers X`: Set the number of peers (default: 4)
- `--use-rag`: Enable RAG for enhanced analysis
- `--use-google`: Use Google Search API for context
- `--json-only`: Output only the JSON result (for programmatic use)

#### Sending Results to Solana

To run the fact-check and send results to Solana:
```
python run_and_send.py --claim "Your claim to fact-check" --sender path/to/keypair.json
```

## Directory Structure

- `anti_misinfo_swarm/`: Core modules for the swarm system
  - `models/`: LLM integrations (OpenAI, Perplexity, etc.)
  - `trainer/`: RL training components
- `configs/`: Configuration files
- `knowledge_base/`: Base knowledge for RAG capability
- `swarm_demo.py`: Main demo script
- `run_and_send.py`: Script to run demo and send to Solana
- `solana_sender.py`: Solana transaction handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI and Perplexity for their language model APIs
- Solana for blockchain infrastructure
- The broader AI and blockchain communities
