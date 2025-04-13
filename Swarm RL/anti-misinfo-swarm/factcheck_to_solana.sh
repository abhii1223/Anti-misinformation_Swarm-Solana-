#!/bin/bash

# Script to fact-check a claim and send the result to Solana blockchain
# Usage: ./factcheck_to_solana.sh "Your claim to check"

# Solana wallet configuration - these can be customized
SENDER_KEYPAIR="/Users/abhimanyugangani/.config/solana/sndUikFkrKdN64Wqg5HTh2ie36qtQKh2zw4gFvXdW9B.json"
RECEIVER_ADDRESS="rcvhHhocNoGETaAQ4GiRtduw12y3Lnp5qAeq9ATaQaP"
LAMPORTS=5000  # Amount to send (5000 lamports = 0.000005 SOL)

# Check if claim is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a claim to fact-check."
    echo "Usage: ./factcheck_to_solana.sh \"Your claim to check\""
    exit 1
fi

CLAIM="$1"
echo "Running fact-check for claim: '$CLAIM'"
echo "==============================================="

# Define the output file for the JSON result
JSON_FILE="/tmp/factcheck_result.json"

# Run the swarm demo with the claim and capture JSON output
python3 swarm_demo.py --claim "$CLAIM" --num-peers 4 --json-only > "$JSON_FILE"

# Check if the fact-check was successful
if [ $? -ne 0 ]; then
    echo "Error: Fact-check process failed."
    exit 1
fi

# Extract verdict from JSON
VERDICT=$(cat "$JSON_FILE" | jq -r .verdict)
echo "Fact-check complete. Result:"
echo "\"$VERDICT\""

echo "==============================================="
echo "Sending result to Solana blockchain..."

# Check if solana_sender.py exists
if [ ! -f "solana_sender.py" ]; then
    echo "Error: solana_sender.py not found."
    exit 1
fi

# Send the result to Solana with the configured keypairs
RESULT=$(python3 solana_sender.py --json-file "$JSON_FILE" --auto-confirm \
    --sender "$SENDER_KEYPAIR" \
    --receiver "$RECEIVER_ADDRESS" \
    --lamports "$LAMPORTS")
EXIT_CODE=$?

# Display the result
echo "$RESULT"

# Check if the transaction was successful
if [ $EXIT_CODE -eq 0 ] && [[ "$RESULT" == *"Transaction successful"* ]]; then
    # Extract transaction signature for reference
    TX_SIGNATURE=$(echo "$RESULT" | grep "Signature:" | awk '{print $2}')
    TX_URL=$(echo "$RESULT" | grep "View on Solana Explorer:" | awk '{print $4}')
    
    echo "✅ Fact-check result successfully recorded on Solana blockchain."
    echo "Transaction ID: $TX_SIGNATURE"
    echo "View transaction: $TX_URL"
else
    echo "❌ Failed to record fact-check result on Solana blockchain."
    exit 1
fi

# Clean up
rm -f "$JSON_FILE"
echo "Done!" 