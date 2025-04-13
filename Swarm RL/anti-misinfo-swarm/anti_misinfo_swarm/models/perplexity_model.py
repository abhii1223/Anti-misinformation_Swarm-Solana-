"""
Perplexity model wrapper for the anti-misinformation swarm.
"""

import json
import logging
import requests
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("perplexity_model")

class PerplexityModel:
    """Wrapper for Perplexity API to provide a consistent interface for the swarm."""
    
    def __init__(self, api_key: str, model_name: str = "sonar", max_tokens: int = 1024, temperature: float = 0.1):
        """
        Initialize the Perplexity model.
        
        Args:
            api_key: Perplexity API key
            model_name: Name of the model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = random)
        """
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Initialized Perplexity model: {model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, json_response: bool = False) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            json_response: Whether to format the response as JSON
            
        Returns:
            The model's response
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        if not system_prompt:
            system_prompt = "You are a helpful, accurate, and unbiased assistant."
            if json_response:
                system_prompt += " Always provide responses in valid JSON format."
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            if response.status_code != 200:
                logger.error(f"API request failed - {response.status_code}")
                if json_response:
                    return json.dumps({"error": f"API request failed - {response.status_code}"})
                return f"Error: API request failed - {response.status_code}"

            result = response.json()
            if 'choices' not in result or len(result['choices']) == 0:
                logger.error("Invalid API response")
                if json_response:
                    return json.dumps({"error": "Invalid API response"})
                return "Error: Invalid API response"

            content = result['choices'][0]['message']['content'].strip()
            
            # If JSON response is requested, ensure it's valid JSON
            if json_response:
                try:
                    # Try to find JSON in the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        # Test if it's valid JSON
                        json.loads(json_str)
                        return json_str
                    else:
                        # Return structured error
                        return json.dumps({"error": "Could not extract JSON from response"})
                except json.JSONDecodeError:
                    # Return structured error
                    return json.dumps({"error": "Invalid JSON in response"})
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating response from Perplexity: {e}")
            if json_response:
                return json.dumps({"error": str(e)})
            return f"Error: {str(e)}"

    def analyze_claim(self, claim: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a claim for misinformation.
        
        Args:
            claim: The claim to analyze
            context: Optional RAG context passages
            
        Returns:
            Analysis results as a dictionary
        """
        context_text = ""
        if context and len(context) > 0:
            context_text = "\n\nRelevant context information:\n" + "\n".join([f"- {c}" for c in context])
        
        system_prompt = """You are an expert fact-checker specializing in detecting misinformation.
Your task is to analyze claims for factual accuracy and provide detailed reasoning.
Base your analysis on facts, logic, and provided context information when available.
Your analysis should be well-structured and formatted in JSON."""
        
        prompt = f"""Analyze the following claim for misinformation:
        
Claim: {claim}
{context_text}

Provide your analysis as a JSON object with the following structure:
{{
    "verdict": "MISINFORMATION" or "ACCURATE" or "UNCERTAIN",
    "confidence": [0.0-1.0 confidence score],
    "reasoning": [detailed explanation for your verdict],
    "evidence": [list of evidence supporting your verdict],
    "limitations": [any limitations in your analysis]
}}
"""
        
        try:
            response = self.generate(prompt, system_prompt, json_response=True)
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from Perplexity")
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "reasoning": "Failed to parse model response",
                "evidence": [],
                "limitations": ["Model response could not be parsed as JSON"]
            }

    def critique_analysis(self, analysis: Dict[str, Any], claim: str) -> Dict[str, Any]:
        """
        Critique another model's analysis of a claim.
        
        Args:
            analysis: Analysis to critique
            claim: The original claim
            
        Returns:
            Critique results as a dictionary
        """
        system_prompt = """You are a critical evaluator specializing in detecting flaws in fact-checking analyses.
Your task is to identify potential issues in another model's analysis of a claim.
Focus on logical errors, unsupported assertions, and potential biases.
Be detailed and specific in your critique, formatting your response in JSON."""
        
        prompt = f"""Review the following analysis of a potentially misleading claim:
        
Claim: {claim}

Analysis:
{json.dumps(analysis, indent=2)}

Provide a critique of this analysis as a JSON object with the following structure:
{{
    "critique_points": [list of specific critique points],
    "missing_considerations": [important factors not considered],
    "accuracy_score": [0.0-1.0 score for the analysis accuracy],
    "bias_assessment": [assessment of potential bias in the analysis],
    "improvement_suggestions": [specific suggestions to improve the analysis]
}}
"""
        
        try:
            response = self.generate(prompt, system_prompt, json_response=True)
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from Perplexity")
            return {
                "critique_points": ["Failed to parse model response"],
                "missing_considerations": [],
                "accuracy_score": 0.0,
                "bias_assessment": "Unknown - parsing error",
                "improvement_suggestions": []
            }

    def form_consensus(self, analyses: List[Dict[str, Any]], critiques: List[Dict[str, Any]], claim: str) -> Dict[str, Any]:
        """
        Form a consensus based on multiple analyses and critiques.
        
        Args:
            analyses: List of analyses from different models
            critiques: List of critiques
            claim: The original claim
            
        Returns:
            Consensus results as a dictionary
        """
        system_prompt = """You are a consensus-building system that integrates multiple analyses and critiques.
Your task is to form a well-reasoned final verdict on a claim by synthesizing different perspectives.
Weight analyses based on their reasonableness, evidence, and how they stand up to critique.
Provide a clear verdict with high-confidence explanations in JSON format."""
        
        prompt = f"""Form a consensus on the following claim based on multiple analyses and critiques:
        
Claim: {claim}

Analyses:
{json.dumps(analyses, indent=2)}

Critiques:
{json.dumps(critiques, indent=2)}

Provide your consensus as a JSON object with the following structure:
{{
    "final_verdict": "MISINFORMATION" or "ACCURATE" or "UNCERTAIN",
    "confidence": [0.0-1.0 confidence score],
    "key_rationale": [main reasons for this verdict],
    "evidence_sources": [key evidence supporting this verdict],
    "disagreements": [notable areas of disagreement between analyses],
    "limitations": [limitations of this consensus]
}}
"""
        
        try:
            response = self.generate(prompt, system_prompt, json_response=True)
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from Perplexity")
            return {
                "final_verdict": "ERROR",
                "confidence": 0.0,
                "key_rationale": ["Failed to parse model response"],
                "evidence_sources": [],
                "disagreements": [],
                "limitations": ["Model response could not be parsed as JSON"]
            } 