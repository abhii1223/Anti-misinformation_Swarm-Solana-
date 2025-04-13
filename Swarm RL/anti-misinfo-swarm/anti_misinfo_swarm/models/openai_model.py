"""
OpenAI model wrapper for the anti-misinformation swarm.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
import openai

logger = logging.getLogger("openai_model")

class OpenAIModel:
    """Wrapper for OpenAI API to provide a consistent interface for the swarm."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o", max_tokens: int = 1024, temperature: float = 0.1):
        """
        Initialize the OpenAI model.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 = deterministic, 1.0 = random)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Initialized OpenAI model: {model_name}")

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
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system = "You are a helpful, accurate, and unbiased assistant."
            if json_response:
                default_system += " Always provide responses in valid JSON format."
            messages.append({"role": "system", "content": default_system})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"} if json_response else None
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
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
        
        system_prompt = """You are an expert fact-checker with deep knowledge of common misinformation.
Your task is to analyze claims for factual accuracy and provide detailed reasoning.
Base your analysis on facts, logic, and provided context information when available.
Ensure your verdict is well-justified and considers multiple perspectives."""
        
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
            logger.error(f"Failed to parse JSON response from OpenAI")
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
Be detailed and specific in your critique."""
        
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
            logger.error(f"Failed to parse JSON response from OpenAI")
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
Provide a clear verdict with high-confidence explanations."""
        
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
            logger.error(f"Failed to parse JSON response from OpenAI")
            return {
                "final_verdict": "ERROR",
                "confidence": 0.0,
                "key_rationale": ["Failed to parse model response"],
                "evidence_sources": [],
                "disagreements": [],
                "limitations": ["Model response could not be parsed as JSON"]
            } 