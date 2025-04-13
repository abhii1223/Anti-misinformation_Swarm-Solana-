"""
Swarm trainer for misinformation detection.

This module implements the three-stage reinforcement learning process:
1. Analysis - Independent assessment of claims
2. Critique - Cross-evaluation of analyses
3. Consensus - Formation of a unified verdict
"""

import json
import time
import logging
import random
import hivemind
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

from ..models.openai_model import OpenAIModel
from ..models.perplexity_model import PerplexityModel
from ..models.rag_retriever import RAGRetriever

logger = logging.getLogger("swarm_trainer")

class SwarmTrainer:
    """
    Implements the three-stage RL training process for misinformation detection.
    """
    
    def __init__(
        self, 
        dht: hivemind.DHT,
        config: Dict[str, Any],
        openai_model: OpenAIModel,
        perplexity_model: PerplexityModel,
        rag_retriever: Optional[RAGRetriever] = None,
        log_tag: str = "unknown_peer"
    ):
        """
        Initialize the swarm trainer.
        
        Args:
            dht: Hivemind DHT for swarm communication
            config: Configuration dictionary
            openai_model: OpenAI model instance
            perplexity_model: Perplexity model instance
            rag_retriever: Optional RAG retriever
            log_tag: Tag for logging
        """
        self.dht = dht
        self.config = config
        self.openai_model = openai_model
        self.perplexity_model = perplexity_model
        self.rag_retriever = rag_retriever
        self.log_tag = log_tag
        
        # Set up DHT keys
        prefix = config.get("swarm", {}).get("dht_prefix", "misinfo_detection")
        self.dht_keys = {
            "claims": f"{prefix}.claims",
            "stage1": f"{prefix}.stage1",
            "stage2": f"{prefix}.stage2",
            "stage3": f"{prefix}.stage3",
            "current_round": f"{prefix}.current_round",
            "active_peers": f"{prefix}.active_peers",
            "rewards": f"{prefix}.rewards"  # Track rewards for each peer
        }
        
        # Initialize stages configuration
        self.stages_config = config.get("stages", {})
        
        # Set training parameters
        self.max_rounds = config.get("swarm", {}).get("max_rounds", 100)
        
        # Last heartbeat time
        self.last_heartbeat = time.time()
        
        # Initialize stats
        self.stats = {
            "analyses_completed": 0,
            "critiques_completed": 0,
            "consensus_completed": 0,
            "total_rewards": 0.0
        }
        
        # Initialize per-stage rewards
        self.reward_weights = {
            "stage1": self.stages_config.get("stage1_analysis", {}).get("reward_weight", 0.4),
            "stage2": self.stages_config.get("stage2_critique", {}).get("reward_weight", 0.3),
            "stage3": self.stages_config.get("stage3_consensus", {}).get("reward_weight", 0.3)
        }
        
        logger.info(f"Initialized SwarmTrainer for peer {log_tag}")
        logger.info(f"Reward weights: Analysis={self.reward_weights['stage1']}, "
                   f"Critique={self.reward_weights['stage2']}, "
                   f"Consensus={self.reward_weights['stage3']}")
    
    def send_heartbeat(self):
        """Register this peer as active in the DHT"""
        try:
            active_peers = self.dht.get(self.dht_keys["active_peers"], latest=True) or {}
            if not isinstance(active_peers, dict):
                active_peers = {}
                
            # Update peer's timestamp
            active_peers[self.log_tag] = time.time()
            
            # Remove peers that haven't sent a heartbeat in the last 5 minutes
            active_peers = {
                peer: timestamp for peer, timestamp in active_peers.items()
                if time.time() - timestamp < 300  # 5 minutes
            }
            
            # Store updated active peers
            self.dht.store(self.dht_keys["active_peers"], active_peers, expiration_time=600)
            self.last_heartbeat = time.time()
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
    
    def get_current_round(self) -> int:
        """Get the current round from the DHT"""
        try:
            round_info = self.dht.get(self.dht_keys["current_round"], latest=True)
            if round_info is None:
                # Initialize to round 0 if not set
                self.dht.store(self.dht_keys["current_round"], 0, expiration_time=24*3600)
                return 0
            return int(round_info)
        except Exception as e:
            logger.error(f"Error getting current round: {e}")
            return 0
    
    def advance_round(self) -> int:
        """Advance to the next round and return the new round number"""
        current_round = self.get_current_round()
        next_round = current_round + 1
        
        try:
            self.dht.store(self.dht_keys["current_round"], next_round, expiration_time=24*3600)
            logger.info(f"Advanced to round {next_round}")
            return next_round
        except Exception as e:
            logger.error(f"Error advancing round: {e}")
            return current_round
    
    def get_active_claims(self) -> List[Dict[str, Any]]:
        """Get the active claims from the DHT"""
        try:
            claims = self.dht.get(self.dht_keys["claims"], latest=True)
            if claims is None:
                # Initialize with sample claims if none exist
                sample_claims = self._get_sample_claims()
                self.dht.store(self.dht_keys["claims"], sample_claims, expiration_time=24*3600)
                return sample_claims
            return claims
        except Exception as e:
            logger.error(f"Error getting active claims: {e}")
            return []
    
    def _get_sample_claims(self) -> List[Dict[str, Any]]:
        """Generate sample claims for training"""
        return [
            {
                "id": "claim_1",
                "text": "COVID-19 vaccines contain microchips to track people."
            },
            {
                "id": "claim_2",
                "text": "Climate change is a hoax created by scientists to get research funding."
            },
            {
                "id": "claim_3",
                "text": "5G networks cause cancer and other health problems."
            },
            {
                "id": "claim_4",
                "text": "The Earth is flat, not spherical."
            },
            {
                "id": "claim_5",
                "text": "Drinking bleach can cure COVID-19."
            }
        ]
    
    def get_stage1_results(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get stage 1 (analysis) results for a claim"""
        stage1_key = f"{self.dht_keys['stage1']}.{claim_id}"
        try:
            results = self.dht.get(stage1_key, latest=True)
            return results or []
        except Exception as e:
            logger.error(f"Error getting stage 1 results: {e}")
            return []
    
    def calculate_stage1_reward(self, analysis: Dict[str, Any]) -> float:
        """Calculate reward for a stage 1 analysis based on confidence and reasoning"""
        reward = 0.0
        
        # Check confidence - higher confidence gets higher reward, but only if correct
        confidence = float(analysis.get("confidence", 0.0))
        
        # Check quality of reasoning
        reasoning = analysis.get("reasoning", "")
        reasoning_length = len(reasoning) if isinstance(reasoning, str) else 0
        evidence = analysis.get("evidence", [])
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
        
        # Basic reward for participation
        reward += 0.1
        
        # Reward for detailed reasoning (up to 0.4)
        reward += min(0.4, reasoning_length / 1000)
        
        # Reward for providing evidence (up to 0.3)
        reward += min(0.3, evidence_count * 0.1)
        
        # Reward for appropriate confidence (up to 0.2)
        if confidence > 0.0 and confidence <= 1.0:
            reward += 0.2 * confidence
        
        return reward * self.reward_weights["stage1"]
    
    def store_stage1_result(self, claim_id: str, result: Dict[str, Any]):
        """Store a stage 1 (analysis) result for a claim"""
        stage1_key = f"{self.dht_keys['stage1']}.{claim_id}"
        try:
            existing_results = self.get_stage1_results(claim_id)
            
            # Calculate reward for this analysis
            reward = self.calculate_stage1_reward(result)
            
            # Add peer identifier and timestamp
            result["peer_id"] = self.log_tag
            result["timestamp"] = time.time()
            result["model"] = "openai" if random.random() < 0.5 else "perplexity"
            result["reward"] = reward
            
            # Add to existing results
            existing_results.append(result)
            
            # Store updated results
            self.dht.store(stage1_key, existing_results, expiration_time=24*3600)
            
            # Update stats
            self.stats["analyses_completed"] += 1
            self.stats["total_rewards"] += reward
            
            # Update rewards in DHT
            self.update_peer_rewards(reward)
            
            logger.info(f"Completed analysis for claim {claim_id}, received reward: {reward:.4f}")
        except Exception as e:
            logger.error(f"Error storing stage 1 result: {e}")
    
    def get_stage2_results(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get stage 2 (critique) results for a claim"""
        stage2_key = f"{self.dht_keys['stage2']}.{claim_id}"
        try:
            results = self.dht.get(stage2_key, latest=True)
            return results or []
        except Exception as e:
            logger.error(f"Error getting stage 2 results: {e}")
            return []
    
    def calculate_stage2_reward(self, critique: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Calculate reward for a stage 2 critique based on quality and insight"""
        reward = 0.0
        
        # Basic reward for participation
        reward += 0.1
        
        # Get critique quality metrics
        critique_points = critique.get("critique_points", [])
        critique_point_count = len(critique_points) if isinstance(critique_points, list) else 0
        
        missing_considerations = critique.get("missing_considerations", [])
        missing_count = len(missing_considerations) if isinstance(missing_considerations, list) else 0
        
        accuracy_score = float(critique.get("accuracy_score", 0.0))
        improvement_suggestions = critique.get("improvement_suggestions", [])
        suggestion_count = len(improvement_suggestions) if isinstance(improvement_suggestions, list) else 0
        
        # Reward for number of critique points (up to 0.3)
        reward += min(0.3, critique_point_count * 0.1)
        
        # Reward for identifying missing considerations (up to 0.2)
        reward += min(0.2, missing_count * 0.1)
        
        # Reward for providing accuracy score aligned with analysis quality
        if 0.0 <= accuracy_score <= 1.0:
            reward += 0.2
        
        # Reward for improvement suggestions (up to 0.2)
        reward += min(0.2, suggestion_count * 0.1)
        
        return reward * self.reward_weights["stage2"]
    
    def store_stage2_result(self, claim_id: str, analysis_peer_id: str, result: Dict[str, Any]):
        """Store a stage 2 (critique) result for a claim"""
        stage2_key = f"{self.dht_keys['stage2']}.{claim_id}"
        try:
            existing_results = self.get_stage2_results(claim_id)
            
            # Get the analysis being critiqued
            analysis = None
            for r in self.get_stage1_results(claim_id):
                if r.get("peer_id") == analysis_peer_id:
                    analysis = r
                    break
            
            # Calculate reward
            reward = 0.0
            if analysis is not None:
                reward = self.calculate_stage2_reward(result, analysis)
            else:
                reward = 0.1 * self.reward_weights["stage2"]  # Base reward only
            
            # Add peer identifier and timestamp
            result["peer_id"] = self.log_tag
            result["analysis_peer_id"] = analysis_peer_id
            result["timestamp"] = time.time()
            result["model"] = "openai" if random.random() < 0.5 else "perplexity"
            result["reward"] = reward
            
            # Add to existing results
            existing_results.append(result)
            
            # Store updated results
            self.dht.store(stage2_key, existing_results, expiration_time=24*3600)
            
            # Update stats
            self.stats["critiques_completed"] += 1
            self.stats["total_rewards"] += reward
            
            # Update rewards in DHT
            self.update_peer_rewards(reward)
            
            logger.info(f"Completed critique for claim {claim_id}, received reward: {reward:.4f}")
        except Exception as e:
            logger.error(f"Error storing stage 2 result: {e}")
    
    def get_stage3_results(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get stage 3 (consensus) results for a claim"""
        stage3_key = f"{self.dht_keys['stage3']}.{claim_id}"
        try:
            results = self.dht.get(stage3_key, latest=True)
            return results or []
        except Exception as e:
            logger.error(f"Error getting stage 3 results: {e}")
            return []
    
    def calculate_stage3_reward(self, consensus: Dict[str, Any], analyses: List[Dict[str, Any]]) -> float:
        """Calculate reward for a stage 3 consensus based on integration and clarity"""
        reward = 0.0
        
        # Basic reward for participation
        reward += 0.1
        
        # Get consensus metrics
        confidence = float(consensus.get("confidence", 0.0))
        key_rationale = consensus.get("key_rationale", [])
        rationale_count = len(key_rationale) if isinstance(key_rationale, list) else 0
        
        evidence_sources = consensus.get("evidence_sources", [])
        evidence_count = len(evidence_sources) if isinstance(evidence_sources, list) else 0
        
        disagreements = consensus.get("disagreements", [])
        disagreement_count = len(disagreements) if isinstance(disagreements, list) else 0
        
        # Check if verdict aligns with most analyses
        verdict_alignment = 0.0
        if analyses:
            verdicts = [a.get("verdict") for a in analyses if "verdict" in a]
            if verdicts and consensus.get("final_verdict") in verdicts:
                majority_verdict = max(set(verdicts), key=verdicts.count)
                if consensus.get("final_verdict") == majority_verdict:
                    verdict_alignment = 0.3
        
        # Reward for confidence (up to 0.1)
        if 0.0 <= confidence <= 1.0:
            reward += 0.1 * confidence
        
        # Reward for key rationale (up to 0.2)
        reward += min(0.2, rationale_count * 0.05)
        
        # Reward for evidence sources (up to 0.2)
        reward += min(0.2, evidence_count * 0.05)
        
        # Reward for acknowledging disagreements (up to 0.1)
        reward += min(0.1, disagreement_count * 0.05)
        
        # Reward for verdict alignment
        reward += verdict_alignment
        
        return reward * self.reward_weights["stage3"]
    
    def store_stage3_result(self, claim_id: str, result: Dict[str, Any]):
        """Store a stage 3 (consensus) result for a claim"""
        stage3_key = f"{self.dht_keys['stage3']}.{claim_id}"
        try:
            existing_results = self.get_stage3_results(claim_id)
            
            # Get all analyses for this claim
            analyses = self.get_stage1_results(claim_id)
            
            # Calculate reward
            reward = self.calculate_stage3_reward(result, analyses)
            
            # Add peer identifier and timestamp
            result["peer_id"] = self.log_tag
            result["timestamp"] = time.time()
            result["model"] = "openai" if random.random() < 0.5 else "perplexity"
            result["reward"] = reward
            
            # Add to existing results
            existing_results.append(result)
            
            # Store updated results
            self.dht.store(stage3_key, existing_results, expiration_time=24*3600)
            
            # Update stats
            self.stats["consensus_completed"] += 1
            self.stats["total_rewards"] += reward
            
            # Update rewards in DHT
            self.update_peer_rewards(reward)
            
            logger.info(f"Completed consensus for claim {claim_id}, received reward: {reward:.4f}")
        except Exception as e:
            logger.error(f"Error storing stage 3 result: {e}")
    
    def update_peer_rewards(self, reward: float):
        """Update the peer's rewards in the DHT"""
        try:
            rewards_key = self.dht_keys["rewards"]
            rewards = self.dht.get(rewards_key, latest=True) or {}
            
            if not isinstance(rewards, dict):
                rewards = {}
            
            # Update or initialize this peer's reward
            if self.log_tag in rewards:
                rewards[self.log_tag] += reward
            else:
                rewards[self.log_tag] = reward
            
            # Store updated rewards
            self.dht.store(rewards_key, rewards, expiration_time=24*3600)
        except Exception as e:
            logger.error(f"Error updating peer rewards: {e}")
    
    def get_all_peer_rewards(self) -> Dict[str, float]:
        """Get rewards for all peers from the DHT"""
        try:
            rewards_key = self.dht_keys["rewards"]
            rewards = self.dht.get(rewards_key, latest=True) or {}
            return rewards if isinstance(rewards, dict) else {}
        except Exception as e:
            logger.error(f"Error getting peer rewards: {e}")
            return {}
    
    def run_stage1_analysis(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Run stage 1 (analysis) on a claim"""
        claim_text = claim["text"]
        claim_id = claim["id"]
        
        # Use RAG if available
        context = []
        if self.rag_retriever:
            context = self.rag_retriever.get_context_passages(claim_text)
        
        # Choose a model randomly (simulates having different models in the swarm)
        if random.random() < 0.5:
            logger.info(f"Analyzing claim using OpenAI: {claim_id}")
            result = self.openai_model.analyze_claim(claim_text, context)
        else:
            logger.info(f"Analyzing claim using Perplexity: {claim_id}")
            result = self.perplexity_model.analyze_claim(claim_text, context)
        
        return result
    
    def run_stage2_critique(self, claim: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run stage 2 (critique) on an analysis"""
        claim_text = claim["text"]
        
        # Choose a model randomly (simulates having different models in the swarm)
        if random.random() < 0.5:
            logger.info(f"Critiquing analysis using OpenAI")
            result = self.openai_model.critique_analysis(analysis, claim_text)
        else:
            logger.info(f"Critiquing analysis using Perplexity")
            result = self.perplexity_model.critique_analysis(analysis, claim_text)
        
        return result
    
    def run_stage3_consensus(self, claim: Dict[str, Any], analyses: List[Dict[str, Any]], critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run stage 3 (consensus) on analyses and critiques"""
        claim_text = claim["text"]
        
        # Choose a model randomly (simulates having different models in the swarm)
        if random.random() < 0.5:
            logger.info(f"Forming consensus using OpenAI")
            result = self.openai_model.form_consensus(analyses, critiques, claim_text)
        else:
            logger.info(f"Forming consensus using Perplexity")
            result = self.perplexity_model.form_consensus(analyses, critiques, claim_text)
        
        return result
    
    def train_step(self):
        """Run a single training step in the current round"""
        # Send heartbeat
        if time.time() - self.last_heartbeat > 60:  # Every minute
            self.send_heartbeat()
        
        # Get current round
        current_round = self.get_current_round()
        
        # Get active claims
        claims = self.get_active_claims()
        if not claims:
            logger.error("No active claims found")
            return
        
        # Pick a random claim to work on
        claim = random.choice(claims)
        claim_id = claim["id"]
        
        # Stage 1: Analysis
        # Check if we've already analyzed this claim in this round
        stage1_results = self.get_stage1_results(claim_id)
        has_analyzed = any(r.get("peer_id") == self.log_tag and r.get("round", 0) == current_round for r in stage1_results)
        
        if not has_analyzed:
            logger.info(f"Running stage 1 (analysis) for claim {claim_id}")
            analysis = self.run_stage1_analysis(claim)
            analysis["round"] = current_round
            self.store_stage1_result(claim_id, analysis)
            time.sleep(1)  # Small delay to avoid hammering the DHT
            return
        
        # Stage 2: Critique
        # Only critique analyses from the current round
        current_round_analyses = [r for r in stage1_results if r.get("round", 0) == current_round]
        
        # Only critique analyses from other peers
        other_peer_analyses = [r for r in current_round_analyses if r.get("peer_id") != self.log_tag]
        
        # Check if there are any analyses to critique
        if not other_peer_analyses:
            # No analyses to critique yet, wait
            logger.info(f"No analyses to critique for claim {claim_id} in round {current_round}")
            time.sleep(2)
            return
        
        # Check existing critiques to avoid duplication
        stage2_results = self.get_stage2_results(claim_id)
        critiqued_peers = set(r.get("analysis_peer_id") for r in stage2_results 
                             if r.get("peer_id") == self.log_tag and r.get("round", 0) == current_round)
        
        uncritiqued_analyses = [r for r in other_peer_analyses if r.get("peer_id") not in critiqued_peers]
        
        if uncritiqued_analyses:
            # Pick a random analysis to critique
            analysis_to_critique = random.choice(uncritiqued_analyses)
            analysis_peer_id = analysis_to_critique.get("peer_id")
            
            logger.info(f"Running stage 2 (critique) for analysis by {analysis_peer_id}")
            critique = self.run_stage2_critique(claim, analysis_to_critique)
            critique["round"] = current_round
            self.store_stage2_result(claim_id, analysis_peer_id, critique)
            time.sleep(1)  # Small delay to avoid hammering the DHT
            return
        
        # Stage 3: Consensus
        # Check if we've already formed a consensus for this claim in this round
        stage3_results = self.get_stage3_results(claim_id)
        has_consensus = any(r.get("peer_id") == self.log_tag and r.get("round", 0) == current_round for r in stage3_results)
        
        if not has_consensus and len(current_round_analyses) >= 2 and len(stage2_results) >= 2:
            logger.info(f"Running stage 3 (consensus) for claim {claim_id}")
            # Filter for critiques from the current round
            current_round_critiques = [r for r in stage2_results if r.get("round", 0) == current_round]
            
            consensus = self.run_stage3_consensus(claim, current_round_analyses, current_round_critiques)
            consensus["round"] = current_round
            self.store_stage3_result(claim_id, consensus)
            time.sleep(1)  # Small delay to avoid hammering the DHT
            return
        
        # If we've done everything for this claim in this round, pick another claim or wait
        logger.info(f"Completed all stages for claim {claim_id} in round {current_round}, waiting...")
        time.sleep(2)
    
    def check_round_complete(self) -> bool:
        """Check if the current round is complete"""
        current_round = self.get_current_round()
        claims = self.get_active_claims()
        
        # Get active peers
        active_peers = self.dht.get(self.dht_keys["active_peers"], latest=True) or {}
        if not isinstance(active_peers, dict):
            active_peers = {}
        
        # Remove peers that haven't sent a heartbeat in the last 5 minutes
        active_peers = {
            peer: timestamp for peer, timestamp in active_peers.items()
            if time.time() - timestamp < 300  # 5 minutes
        }
        
        active_peer_count = len(active_peers)
        if active_peer_count == 0:
            return False  # No active peers, cannot complete round
        
        # Check if each claim has consensus from at least half of the active peers
        for claim in claims:
            claim_id = claim["id"]
            stage3_results = self.get_stage3_results(claim_id)
            current_round_consensus = [r for r in stage3_results if r.get("round", 0) == current_round]
            
            if len(current_round_consensus) < max(1, active_peer_count // 2):
                return False  # Not enough consensus for this claim
        
        return True  # All claims have sufficient consensus
    
    def show_reward_leaderboard(self):
        """Display the reward leaderboard for all peers"""
        rewards = self.get_all_peer_rewards()
        if not rewards:
            logger.info("No rewards recorded yet")
            return
        
        # Sort peers by rewards
        sorted_peers = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("===== Reward Leaderboard =====")
        for i, (peer, reward) in enumerate(sorted_peers):
            logger.info(f"#{i+1}: {peer} - {reward:.4f} points")
        logger.info("==============================")
    
    def train(self):
        """Run the full training process"""
        logger.info(f"Starting training with peer ID {self.log_tag}")
        logger.info(f"Reward structure: Analysis={self.reward_weights['stage1']:.2f}, "
                   f"Critique={self.reward_weights['stage2']:.2f}, "
                   f"Consensus={self.reward_weights['stage3']:.2f}")
        
        try:
            round_count = 0
            while round_count < self.max_rounds:
                current_round = self.get_current_round()
                logger.info(f"Training round {current_round}")
                
                # Run training steps until the round is complete
                steps_in_round = 0
                while steps_in_round < 100 and not self.check_round_complete():  # Max 100 steps per round
                    self.train_step()
                    steps_in_round += 1
                
                # Show reward leaderboard at the end of each round
                self.show_reward_leaderboard()
                
                # Advance to the next round if this round is complete
                if self.check_round_complete():
                    logger.info(f"Round {current_round} complete")
                    self.advance_round()
                    round_count += 1
                else:
                    # Wait a bit before checking again
                    logger.info(f"Round {current_round} not yet complete, waiting...")
                    time.sleep(10)
                
                # Log progress
                logger.info(f"Training stats: {self.stats}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        except Exception as e:
            logger.error(f"Error during training: {e}")
        finally:
            # Final leaderboard
            self.show_reward_leaderboard()
            logger.info(f"Training ended. Final stats: {self.stats}") 