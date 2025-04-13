# Installation Guide

This document provides detailed installation and setup instructions for the Anti-Misinformation Swarm RL on Solana project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Solana CLI tools (for blockchain integration)
- Node.js and npm (if using the web interface)

## Step 1: Clone the Repository

```bash
git clone https://github.com/abhii1223/Anti-misinformation_Swarm-Solana-.git
cd Anti-misinformation_Swarm-Solana-
```

## Step 2: Set Up Python Environment

It's recommended to use a virtual environment:

```bash
cd "Swarm RL/anti-misinfo-swarm"
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Configure API Keys

1. Create an API keys file:
   ```bash
   cp api_keys.example.json api_keys.json
   ```

2. Edit `api_keys.json` and add your actual API keys:
   - OpenAI API key (required)
   - Perplexity API key (optional but recommended)
   - Google API key and Custom Search Engine ID (optional, for web search capabilities)

## Step 5: Set Up Solana (for blockchain integration)

1. Install Solana CLI tools following the [official guide](https://docs.solana.com/cli/install-solana-cli-tools)

2. Create a Solana keypair if you don't have one:
   ```bash
   solana-keygen new --outfile ~/.config/solana/id.json
   ```

3. Configure Solana to use the devnet (for testing):
   ```bash
   solana config set --url https://api.devnet.solana.com
   ```

4. Request airdrop to fund your account (for devnet testing):
   ```bash
   solana airdrop 2
   ```

## Step 6: Initialize Knowledge Base (for RAG capability)

If you plan to use the RAG (Retrieval-Augmented Generation) feature:

```bash
python init_knowledge_base.py
```

This will create a knowledge base with factual information that the system can use for enhanced fact-checking.

## Step 7: Test the System

Run a basic test to ensure everything is working:

```bash
python swarm_demo.py --claim "Earth is the third planet from the Sun"
```

## Troubleshooting

- **API Key Issues**: If you encounter errors about invalid API keys, double-check your `api_keys.json` file
- **Solana Connection Errors**: Ensure your Solana CLI is properly configured and connected to the desired network
- **Python Package Errors**: Make sure you're using the virtual environment and have installed all dependencies

## Next Steps

Refer to the [README.md](README.md) for information on how to use the system's various features and commands. 