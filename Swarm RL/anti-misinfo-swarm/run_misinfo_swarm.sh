#!/bin/bash

# General arguments
ROOT=$PWD

export LOG_LEVEL=INFO
export API_KEYS_FILE="$ROOT/api_keys.json"

GREEN_TEXT="\033[32m"
RESET_TEXT="\033[0m"

echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

# Check if we have API keys file
if [ ! -f "$API_KEYS_FILE" ]; then
    echo_green ">> API keys file not found. Creating one..."
    echo "{
        \"openai\": \"YOUR_OPENAI_API_KEY\",
        \"perplexity\": \"YOUR_PERPLEXITY_API_KEY\"
    }" > "$API_KEYS_FILE"
    echo_green ">> Please edit $API_KEYS_FILE and add your API keys."
    exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo_green ">> Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo_green ">> Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo_green ">> Installing requirements..."
pip install -q -r "$ROOT/requirements.txt"

# Choose a model configuration
echo_green ">> Selecting model configuration..."
if [ -z "$MODEL_CONFIG" ]; then
    CONFIG_PATH="$ROOT/configs/default_config.yaml"
else
    CONFIG_PATH="$ROOT/configs/$MODEL_CONFIG"
fi

echo_green ">> Using config file: $CONFIG_PATH"

# Initialize the knowledge base if it doesn't exist
if [ ! -d "$ROOT/knowledge_base" ]; then
    echo_green ">> Initializing knowledge base..."
    python -m anti_misinfo_swarm.init_knowledge_base
fi

# Run the swarm
echo_green ">> Starting misinformation detection swarm..."
python -m anti_misinfo_swarm.train_swarm --config "$CONFIG_PATH"

# Keep the script running
echo_green ">> Swarm is running. Press Ctrl+C to stop."
wait 