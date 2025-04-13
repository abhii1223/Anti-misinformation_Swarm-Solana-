# Anti-Misinformation Swarm

A decentralized system for detecting text-based misinformation using Swarm Reinforcement Learning with both OpenAI and Perplexity models.

## Overview

This project implements a peer-to-peer network of AI models that collaborate to detect misinformation through a three-stage reinforcement learning process:

1. **Analysis**: Agents independently assess claims for factual accuracy
2. **Critique**: Agents review each other's analyses to identify flaws and insights
3. **Consensus**: Agents converge on a unified verdict by weighing all evidence

The system uses a combination of OpenAI and Perplexity models and can leverage either Retrieval-Augmented Generation (RAG) or Google Search API to ground responses in trusted external sources.

## Features

- **Multi-model swarm**: Uses both OpenAI (gpt-4o) and Perplexity (sonar) models for diverse perspectives
- **Three-stage RL**: Analysis → Critique → Consensus pipeline with weighted rewards
- **Peer-to-peer architecture**: Built with Hivemind for distributed training
- **Context providers**: Optional knowledge grounding using:
  - **RAG**: Local knowledge base for evidence-backed reasoning
  - **Google Search API**: Real-time web search for up-to-date information
- **Explainable outputs**: Detailed analyses with evidence and confidence scores
- **Distributed rewards**: Performance-based incentive system that encourages accuracy

## Reward Mechanism

The system implements a detailed reward mechanism:

| Stage | Weight | Reward Criteria |
|-------|--------|-----------------|
| Analysis | 40% | - Factual grounding<br>- Evidence quality<br>- Reasoning depth<br>- Appropriate confidence |
| Critique | 30% | - Insightful critique points<br>- Identifying missing considerations<br>- Accuracy scoring<br>- Constructive suggestions |
| Consensus | 30% | - Evidence integration<br>- Source citation<br>- Acknowledging disagreements<br>- Verdict alignment |

This reward structure encourages:
- Thorough, evidence-based analysis
- Critical evaluation of other agents' work
- Balanced consensus-building that integrates multiple perspectives

## Requirements

- Python 3.10+
- OpenAI API key
- Perplexity API key
- Optional: Google API key and Custom Search Engine ID
- Minimum 8GB RAM

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/anti-misinfo-swarm.git
   cd anti-misinfo-swarm
   ```

2. Run the setup script (this will create a virtual environment and install dependencies):
   ```
   ./run_misinfo_swarm.sh
   ```

3. The script will create an `api_keys.json` file. Edit this file to add your API keys:
   ```json
   {
       "openai": "YOUR_OPENAI_API_KEY",
       "perplexity": "YOUR_PERPLEXITY_API_KEY",
       "google": {
           "api_key": "YOUR_GOOGLE_API_KEY",
           "cse_id": "YOUR_CUSTOM_SEARCH_ENGINE_ID"
       }
   }
   ```

   For Google Search integration:
   - Create a [Google Cloud Project](https://console.cloud.google.com/)
   - Enable the [Custom Search JSON API](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)
   - Create an [API key](https://console.cloud.google.com/apis/credentials)
   - Create a [Custom Search Engine](https://programmablesearchengine.google.com/cse/all) and get the CSE ID

## Running the Enhanced Demo

The enhanced demo visualizes the swarm architecture, showing multiple peers collaborating with rewards:

```bash
# Basic demo (using local knowledge base):
python swarm_demo.py --use-rag

# Demo using Google Search for context:
python swarm_demo.py --use-google

# Change the number of peers:
python swarm_demo.py --use-rag --num-peers 6
```

## Running the Simple Demo

To test the system without the full swarm or visualization:

```
./demo.py
```

Add the `--use-rag` flag to enable the RAG component:

```
./demo.py --use-rag
```

## Running the Swarm

To run the full swarm (distributed system with multiple peers):

```
./run_misinfo_swarm.sh
```

You can customize the configuration by editing `configs/default_config.yaml`.

## Project Structure

- `anti_misinfo_swarm/` - Main package
  - `models/` - Model implementations
    - `openai_model.py` - OpenAI API wrapper
    - `perplexity_model.py` - Perplexity API wrapper
    - `rag_retriever.py` - RAG component
    - `google_context_provider.py` - Google Search integration
  - `trainer/` - Swarm training implementation
    - `swarm_trainer.py` - Main trainer class with reward mechanism
  - `init_knowledge_base.py` - Knowledge base initialization
  - `train_swarm.py` - Main training script
- `configs/` - Configuration files
  - `default_config.yaml` - Default configuration
- `knowledge_base/` - RAG knowledge base (created at runtime)
- `run_misinfo_swarm.sh` - Main startup script
- `demo.py` - Simple demo script
- `swarm_demo.py` - Enhanced demo with visualization

## How It Works

1. **Peer-to-peer network**: Nodes connect using the Hivemind DHT
2. **Claim analysis**: Each peer analyzes claims using either OpenAI or Perplexity
3. **Cross-critique**: Peers critique each other's analyses
4. **Consensus formation**: Peers form a collective verdict
5. **Reinforcement learning**: The system improves over time through rewards and penalties
6. **Leaderboard tracking**: Peers' performance is tracked and displayed

## Configuration

The system can be configured by editing `configs/default_config.yaml`. Key parameters include:

- Model settings (temperature, token limits)
- Swarm parameters (peer count, rounds)
- RAG settings (retrieval count, embedding model)
- Stage configurations (weights, prompt templates)
- Reward values for different criteria

## Extending the System

This system can be extended in several ways:

- Add additional models beyond OpenAI and Perplexity
- Enhance the RAG knowledge base with more trusted sources
- Customize reward functions for domain-specific accuracy
- Add more sophisticated metrics for evaluating claims
- Implement a web interface for easier interaction

## License

[MIT License](LICENSE)

## Acknowledgments

This project is inspired by the [RL-Swarm](https://github.com/gensyn/rl-swarm) project from Gensyn and builds on their peer-to-peer reinforcement learning architecture. 