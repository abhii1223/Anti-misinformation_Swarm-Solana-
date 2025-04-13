# Installation Guide

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Solana CLI tools
- OpenAI API key
- Perplexity API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhii1223/Anti-misinformation_Swarm-Solana-.git
   cd Anti-misinformation_Swarm-Solana-
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Install backend dependencies:
   ```bash
   cd "Swarm RL/anti-misinfo-swarm"
   pip install -r requirements.txt
   ```

4. Configure API keys:
   ```bash
   # Create the API keys file with your credentials
   cp "Swarm RL/anti-misinfo-swarm/api_keys.example.json" "Swarm RL/anti-misinfo-swarm/api_keys.json"
   # Edit the file with your own API keys
   nano "Swarm RL/anti-misinfo-swarm/api_keys.json"
   ```

5. Set up Solana wallet:
   ```bash
   # Create a new Solana keypair or import an existing one
   solana-keygen new -o ~/.config/solana/factcheck-wallet.json
   ```

## Configuration

1. Update the sender and receiver addresses in `factcheck_to_solana.sh`:
   ```bash
   SENDER_KEYPAIR="/path/to/your/wallet.json"
   RECEIVER_ADDRESS="your-receiver-address"
   ```

2. Update the receiver address in `src/components/solana/transactions-list.tsx`:
   ```typescript
   const RECEIVER_ADDRESS = 'your-receiver-address';
   ```

## Running the Application

1. Start the frontend development server:
   ```bash
   npm run dev
   ```

2. Access the application at http://localhost:3000

## Factchecking a Claim

You can verify claims in two ways:

1. **Through the Web Interface**:
   - Enter a claim in the text area on the homepage
   - Click "Check Claim" to submit
   - View the verification result and blockchain transaction

2. **Using the Command Line**:
   ```bash
   cd "Swarm RL/anti-misinfo-swarm"
   ./factcheck_to_solana.sh "The claim to verify"
   ``` 