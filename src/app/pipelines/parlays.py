"""Parlay optimization pipeline."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from itertools import combinations

from ..config import settings
from ..logging import get_logger
from ..utils.betting import american_to_prob, kelly_fraction, calculate_ev

logger = get_logger(__name__)


class ParlayOptimizer:
    """Optimizer for creating profitable parlays."""
    
    def __init__(self):
        """Initialize parlay optimizer."""
        self.artifacts_dir = settings.artifacts_dir
        self.artifacts_dir.mkdir(exist_ok=True)
        self.max_kelly_fraction = settings.max_kelly_fraction
    
    def calculate_correlation_matrix(self, predictions_df: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix between bets."""
        logger.info("Calculating correlation matrix")
        
        # This is a simplified correlation calculation
        # In production, you'd use historical data to calculate actual correlations
        
        n_bets = len(predictions_df)
        correlation_matrix = np.eye(n_bets)  # Start with identity matrix
        
        # Add some correlation based on team overlap, market type, etc.
        for i in range(n_bets):
            for j in range(i + 1, n_bets):
                # Simplified correlation logic
                bet_i = predictions_df.iloc[i]
                bet_j = predictions_df.iloc[j]
                
                correlation = 0.0
                
                # Same team correlation
                if (bet_i.get('home_team') == bet_j.get('home_team') or 
                    bet_i.get('away_team') == bet_j.get('away_team')):
                    correlation += 0.3
                
                # Same market type correlation
                if bet_i.get('market') == bet_j.get('market'):
                    correlation += 0.2
                
                # Same period correlation
                if bet_i.get('period') == bet_j.get('period'):
                    correlation += 0.1
                
                # Cap correlation
                correlation = min(correlation, 0.8)
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        logger.info(f"Calculated correlation matrix for {n_bets} bets")
        return correlation_matrix
    
    def calculate_parlay_probability(self, bet_indices: List[int], predictions_df: pd.DataFrame, 
                                   correlation_matrix: np.ndarray) -> float:
        """Calculate parlay probability with correlation adjustment."""
        if len(bet_indices) == 1:
            return predictions_df.iloc[bet_indices[0]]['model_prob']
        
        # Get individual probabilities
        probs = [predictions_df.iloc[i]['model_prob'] for i in bet_indices]
        
        # Calculate correlation penalty
        correlation_penalty = 0.0
        for i in range(len(bet_indices)):
            for j in range(i + 1, len(bet_indices)):
                idx_i, idx_j = bet_indices[i], bet_indices[j]
                correlation = correlation_matrix[idx_i, idx_j]
                correlation_penalty += correlation * 0.1  # Penalty factor
        
        # Adjust probabilities
        adjusted_probs = [p * (1 - correlation_penalty) for p in probs]
        
        # Calculate parlay probability (assuming some independence)
        parlay_prob = np.prod(adjusted_probs)
        
        return parlay_prob
    
    def calculate_parlay_odds(self, bet_indices: List[int], predictions_df: pd.DataFrame) -> float:
        """Calculate parlay odds."""
        decimal_odds = [predictions_df.iloc[i]['decimal_odds'] for i in bet_indices]
        parlay_odds = np.prod(decimal_odds)
        return parlay_odds
    
    def evaluate_parlay(self, bet_indices: List[int], predictions_df: pd.DataFrame, 
                       correlation_matrix: np.ndarray) -> Dict:
        """Evaluate a parlay combination."""
        # Calculate parlay probability
        parlay_prob = self.calculate_parlay_probability(bet_indices, predictions_df, correlation_matrix)
        
        # Calculate parlay odds
        parlay_odds = self.calculate_parlay_odds(bet_indices, predictions_df)
        
        # Calculate expected value
        ev = calculate_ev(parlay_prob, parlay_odds)
        
        # Calculate Kelly fraction
        kelly_frac = min(kelly_fraction(parlay_prob, parlay_odds), self.max_kelly_fraction)
        
        # Get bet details
        bet_details = []
        for idx in bet_indices:
            bet = predictions_df.iloc[idx]
            bet_details.append({
                'game_id': bet['game_id'],
                'market': bet.get('market', 'unknown'),
                'period': bet.get('period', 'game'),
                'model_prob': bet['model_prob'],
                'decimal_odds': bet['decimal_odds'],
            })
        
        return {
            'bet_indices': bet_indices,
            'parlay_prob': parlay_prob,
            'parlay_odds': parlay_odds,
            'ev': ev,
            'kelly_fraction': kelly_frac,
            'bet_details': bet_details,
        }
    
    def generate_parlay_combinations(self, predictions_df: pd.DataFrame, 
                                   max_legs: int = 3, top_n: int = 20) -> List[Dict]:
        """Generate top parlay combinations."""
        logger.info(f"Generating parlay combinations (max {max_legs} legs, top {top_n})")
        
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(predictions_df)
        
        # Generate all possible combinations
        all_combinations = []
        
        for r in range(2, max_legs + 1):  # Start from 2-leg parlays
            for combo in combinations(range(len(predictions_df)), r):
                parlay_eval = self.evaluate_parlay(list(combo), predictions_df, correlation_matrix)
                all_combinations.append(parlay_eval)
        
        # Sort by expected value
        all_combinations.sort(key=lambda x: x['ev'], reverse=True)
        
        # Return top N
        top_combinations = all_combinations[:top_n]
        
        logger.info(f"Generated {len(top_combinations)} top parlay combinations")
        return top_combinations
    
    def create_parlay_recommendations(self, date: str, top_n: int = 20, 
                                   risk_percentage: float = 1.0) -> List[Dict]:
        """Create parlay recommendations for a specific date."""
        logger.info(f"Creating parlay recommendations for {date}")
        
        # Load model predictions (simplified - would load actual predictions)
        # For now, create simulated data
        np.random.seed(settings.random_seed)
        n_games = 50
        
        predictions_df = pd.DataFrame({
            'game_id': [f'game_{i}' for i in range(n_games)],
            'model_prob': np.random.beta(2, 2, n_games),
            'market_prob': np.random.beta(2, 2, n_games),
            'american_odds': np.random.choice([-110, -105, -115, 110, 105, 115], n_games),
            'decimal_odds': np.random.choice([1.91, 1.95, 1.87, 2.10, 2.05, 2.15], n_games),
            'market': np.random.choice(['spread', 'total'], n_games),
            'period': np.random.choice(['game', '1H', '1Q'], n_games),
            'home_team': [f'Team_{i}' for i in range(n_games)],
            'away_team': [f'Team_{i+1}' for i in range(n_games)],
        })
        
        # Filter for positive EV bets
        positive_ev_bets = predictions_df[
            predictions_df['model_prob'] > predictions_df['market_prob']
        ].copy()
        
        if len(positive_ev_bets) < 2:
            logger.warning("Not enough positive EV bets for parlays")
            return []
        
        # Generate parlay combinations
        parlay_combinations = self.generate_parlay_combinations(
            positive_ev_bets, max_legs=3, top_n=top_n
        )
        
        # Add risk sizing
        bankroll = 1000  # Assume $1000 bankroll
        risk_amount = bankroll * (risk_percentage / 100)
        
        recommendations = []
        for i, parlay in enumerate(parlay_combinations):
            if parlay['ev'] > 0:  # Only recommend positive EV parlays
                stake = min(risk_amount * parlay['kelly_fraction'], risk_amount * 0.1)  # Cap at 10% of risk
                
                recommendation = {
                    'parlay_id': f"parlay_{i+1}",
                    'date': date,
                    'legs': len(parlay['bet_indices']),
                    'parlay_prob': parlay['parlay_prob'],
                    'parlay_odds': parlay['parlay_odds'],
                    'ev': parlay['ev'],
                    'kelly_fraction': parlay['kelly_fraction'],
                    'recommended_stake': stake,
                    'potential_return': stake * (parlay['parlay_odds'] - 1),
                    'bets': parlay['bet_details'],
                }
                
                recommendations.append(recommendation)
        
        logger.info(f"Created {len(recommendations)} parlay recommendations")
        return recommendations
    
    def save_parlay_recommendations(self, recommendations: List[Dict], date: str) -> None:
        """Save parlay recommendations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        if recommendations:
            df = pd.DataFrame(recommendations)
            csv_filename = f"parlay_recommendations_{date}_{timestamp}.csv"
            csv_path = self.artifacts_dir / csv_filename
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved parlay recommendations to {csv_path}")
        
        # Save as pickle for programmatic access
        pkl_filename = f"parlay_recommendations_{date}_{timestamp}.pkl"
        pkl_path = self.artifacts_dir / pkl_filename
        
        import pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(recommendations, f)
        
        logger.info(f"Saved parlay recommendations to {pkl_path}")
    
    def run_parlay_optimization(self, date: str, top_n: int = 20, 
                               risk_percentage: float = 1.0) -> List[Dict]:
        """Run full parlay optimization."""
        logger.info(f"Running parlay optimization for {date}")
        
        # Create recommendations
        recommendations = self.create_parlay_recommendations(date, top_n, risk_percentage)
        
        # Save recommendations
        self.save_parlay_recommendations(recommendations, date)
        
        logger.info(f"Parlay optimization completed. Generated {len(recommendations)} recommendations")
        return recommendations
