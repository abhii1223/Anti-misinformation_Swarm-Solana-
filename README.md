# Swarm RL Fact Checker

A decentralized fact-checking platform that uses AI swarm intelligence to verify claims and records the results on the Solana blockchain.

## Project Overview

The Swarm RL Fact Checker uses a collaborative network of AI models (a "swarm") to analyze claims and produce a consensus verdict on their accuracy. Each verification is permanently recorded on the Solana blockchain for transparency and immutability.

### Key Features

- **AI Swarm Analysis**: Multiple AI models analyze claims independently and cross-critique each other's work
- **Blockchain Verification**: All fact-check results are recorded on Solana as on-chain transactions
- **Real-time Updates**: Submit claims and view results in a modern, responsive interface
- **Transparent History**: Browse all previous fact-checks with full transaction details

## Architecture

The system consists of three main components:

1. **Next.js Frontend**: A modern web interface for submitting claims and viewing results
2. **Swarm RL Backend**: Python-based AI swarm that analyzes claims using OpenAI and Perplexity models
3. **Solana Integration**: Records verification results on the Solana blockchain for permanence

## Getting Started

See the [Installation Guide](INSTALL.md) for setup instructions.
