#!/usr/bin/env python3
"""
Initialize a small knowledge base for the anti-misinformation swarm demo.
"""

import os
import json
import sys
from anti_misinfo_swarm.init_knowledge_base import KnowledgeBaseInitializer

def main():
    """Initialize the knowledge base."""
    print("Initializing knowledge base...")
    
    # Initialize the knowledge base
    initializer = KnowledgeBaseInitializer()
    initializer.initialize()
    
    print("Knowledge base initialization complete!")

if __name__ == "__main__":
    main() 