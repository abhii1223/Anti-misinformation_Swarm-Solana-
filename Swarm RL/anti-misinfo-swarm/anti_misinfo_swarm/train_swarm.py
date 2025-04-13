"""
Main training script for the anti-misinformation swarm.

This script sets up the DHT network, initializes the models,
and starts the training process.
"""

import os
import json
import yaml
import logging
import argparse
import hivemind
from typing import Dict, Any
from pathlib import Path

from .models.openai_model import OpenAIModel
from .models.perplexity_model import PerplexityModel
from .models.rag_retriever import load_retriever
from .trainer.swarm_trainer import SwarmTrainer

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_swarm")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_api_keys(api_keys_path: str) -> Dict[str, str]:
    """Load API keys from JSON file"""
    with open(api_keys_path, 'r') as f:
        api_keys = json.load(f)
    return api_keys

def setup_dht(config: Dict[str, Any]) -> (hivemind.DHT, str):
    """Set up DHT network and return DHT client and peer ID"""
    swarm_config = config.get("swarm", {})
    
    # Set up DHT arguments
    dht_kwargs = {}
    
    # Add initial peers if provided
    initial_peers = swarm_config.get("initial_peers", [])
    if initial_peers:
        dht_kwargs["initial_peers"] = initial_peers
    
    # Add host address if provided
    host_maddr = swarm_config.get("host_maddr")
    if host_maddr:
        dht_kwargs["host_maddrs"] = [host_maddr]
    
    # Create DHT client
    dht = hivemind.DHT(start=True, **dht_kwargs)
    
    # Get peer ID
    peer_id = str(dht.peer_id)
    peer_tag = f"peer_{peer_id[:8]}"  # Shortened peer ID as tag
    
    logger.info(f"Initialized DHT client with peer ID: {peer_id}")
    logger.info(f"Your swarm peer tag is: {peer_tag}")
    
    if initial_peers:
        logger.info(f"Joining swarm with initial peers: {initial_peers}")
    else:
        logger.info(f"Starting new swarm at {host_maddr}")
    
    return dht, peer_tag

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Anti-Misinformation Swarm Training")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--api-keys", type=str, default="api_keys.json",
                        help="Path to API keys file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Load API keys
    api_keys = load_api_keys(args.api_keys)
    logger.info(f"Loaded API keys from {args.api_keys}")
    
    # Set up DHT and get peer ID
    dht, peer_tag = setup_dht(config)
    
    # Initialize models
    openai_config = config.get("openai_model", {})
    openai_model = OpenAIModel(
        api_key=api_keys.get("openai", ""),
        model_name=openai_config.get("model_name", "gpt-4o"),
        max_tokens=openai_config.get("max_tokens", 1024),
        temperature=openai_config.get("temperature", 0.1)
    )
    
    perplexity_config = config.get("perplexity_model", {})
    perplexity_model = PerplexityModel(
        api_key=api_keys.get("perplexity", ""),
        model_name=perplexity_config.get("model_name", "sonar"),
        max_tokens=perplexity_config.get("max_tokens", 1024),
        temperature=perplexity_config.get("temperature", 0.1)
    )
    
    # Initialize RAG retriever if enabled
    rag_config = config.get("rag", {})
    rag_retriever = load_retriever(rag_config)
    
    # Initialize trainer
    trainer = SwarmTrainer(
        dht=dht,
        config=config,
        openai_model=openai_model,
        perplexity_model=perplexity_model,
        rag_retriever=rag_retriever,
        log_tag=peer_tag
    )
    
    # Start training
    logger.info("Starting swarm training...")
    trainer.train()

if __name__ == "__main__":
    main() 