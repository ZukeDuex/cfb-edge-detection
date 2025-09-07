#!/usr/bin/env python3
"""
Sophisticated CFB Betting Model - Leveraging Organized Analysis
This model builds upon all previous analysis to create an advanced betting system.
"""

import pandas as pd
import numpy as np
import sys
import os
# Add parent directory to path for imports
sys.path.append('../../src')
try:
    from app.config import settings
    API_KEY = settings.cfbd_api_key
except:
    # Fallback if config not available
    API_KEY = "your_api_key_here"
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier
from sklearn.linear_model import Ridge, LogisticRegression, BayesianRidge, ElasticNet, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

class SophisticatedCFBModel:
    """
    Advanced CFB Betting Model that leverages all previous analysis.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_metrics = {}
        self.betting_strategy = {}
        
    def load_organized_data(self):
        """Load all organized data from the analysis folder."""
        print("📁 Loading organized analysis data...")
        
        try:
            # Load core data
            self.games_df = pd.read_csv('analysis/data/tennessee_games_2022_2024.csv')
            self.odds_df = pd.read_csv('analysis/data/tennessee_odds_2022_2024.csv')
            self.enhanced_df = pd.read_csv('analysis/data/tennessee_games_error_minimized.csv')
            self.betting_df = pd.read_csv('analysis/data/tennessee_games_profitable_betting.csv')
            
            # Load results
            self.betting_results = pd.read_csv('analysis/results/simple_betting_analysis.csv')
            self.strategy_summary = pd.read_csv('analysis/results/betting_strategy_summary.csv')
            
            print(f"✅ Loaded {len(self.games_df)} games, {len(self.odds_df)} betting lines")
            print(f"✅ Enhanced features: {len(self.enhanced_df.columns)} columns")
            print(f"✅ Betting results: {len(self.betting_results)} opportunities")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def create_advanced_features(self):
        """Create advanced features based on previous analysis insights."""
        print("🔧 Creating advanced features...")
        
        # Start with enhanced features
        self.advanced_df = self.enhanced_df.copy()
        
        # 1. Betting-specific features based on analysis
        self.advanced_df['is_profitable_opponent'] = self.advanced_df['awayTeam'].isin([
            'Alabama', 'Vanderbilt', 'Georgia', 'Akron', 'UTEP'
        ])
        self.advanced_df['is_avoid_opponent'] = self.advanced_df['awayTeam'].isin([
            'Florida', 'Clemson', 'Arkansas'
        ])
        
        # 2. Edge-based features
        self.advanced_df['predicted_edge'] = self.advanced_df['tennessee_point_differential'] - 7.5  # Default line
        self.advanced_df['edge_category'] = pd.cut(
            self.advanced_df['predicted_edge'].abs(),
            bins=[0, 3, 10, 20, 100],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        # 3. Season momentum features
        self.advanced_df = self.advanced_df.sort_values(['season', 'week'])
        self.advanced_df['season_momentum'] = self.advanced_df.groupby('season')['tennessee_point_differential'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        self.advanced_df['season_consistency'] = self.advanced_df.groupby('season')['tennessee_point_differential'].rolling(window=5, min_periods=1).std().reset_index(0, drop=True)
        
        # 4. Opponent strength features
        opponent_strength = {
            'Alabama': 0.95, 'Georgia': 0.95, 'Ohio State': 0.90, 'Clemson': 0.90,
            'LSU': 0.85, 'Florida': 0.80, 'Auburn': 0.80, 'Texas A&M': 0.75,
            'Kentucky': 0.70, 'South Carolina': 0.70, 'Missouri': 0.65,
            'Vanderbilt': 0.30, 'Akron': 0.20, 'UTEP': 0.15, 'Chattanooga': 0.10
        }
        self.advanced_df['opponent_strength'] = self.advanced_df['awayTeam'].map(opponent_strength).fillna(0.50)
        
        # 5. Betting confidence features
        self.advanced_df['betting_confidence'] = np.where(
            self.advanced_df['predicted_edge'].abs() > 10, 'High',
            np.where(self.advanced_df['predicted_edge'].abs() > 5, 'Medium', 'Low')
        )
        
        # 6. Historical performance features
        self.advanced_df['historical_vs_opponent'] = self.advanced_df.groupby('awayTeam')['tennessee_point_differential'].transform('mean')
        self.advanced_df['historical_win_rate_vs_opponent'] = self.advanced_df.groupby('awayTeam')['tennessee_won'].transform('mean')
        
        print(f"✅ Created {len(self.advanced_df.columns) - len(self.enhanced_df.columns)} advanced features")
        
        return self.advanced_df
    
    def build_sophisticated_models(self):
        """Build sophisticated ML models using all available features."""
        print("🤖 Building sophisticated ML models...")
        
        # Prepare features
        feature_columns = [
            'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
            'homePregameElo', 'awayPregameElo', 'homePostgameWinProbability', 'awayPostgameWinProbability',
            'excitement_index', 'opponent_strength', 'is_profitable_opponent', 'is_avoid_opponent',
            'predicted_edge', 'season_momentum', 'season_consistency', 'betting_confidence',
            'historical_vs_opponent', 'historical_win_rate_vs_opponent'
        ]
        
        # Add all enhanced features
        enhanced_features = [col for col in self.advanced_df.columns if col not in feature_columns and col not in ['id', 'season', 'seasonType', 'startDate', 'homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 'tennessee_won', 'tennessee_point_differential']]
        feature_columns.extend(enhanced_features)
        
        # Filter to available features
        available_features = [f for f in feature_columns if f in self.advanced_df.columns]
        
        print(f"📊 Using {len(available_features)} features for sophisticated modeling")
        
        # Split data
        train_df = self.advanced_df[self.advanced_df['season'].isin([2022, 2023])].copy()
        test_df = self.advanced_df[self.advanced_df['season'] == 2024].copy()
        
        if len(test_df) == 0:
            print("❌ No 2024 data for testing")
            return False
        
        # Prepare data
        X_train = train_df[available_features].fillna(0)
        y_train_reg = train_df['tennessee_point_differential']
        y_train_clf = train_df['tennessee_won']
        
        X_test = test_df[available_features].fillna(0)
        y_test_reg = test_df['tennessee_point_differential']
        y_test_clf = test_df['tennessee_won']
        
        # Advanced feature selection
        print("🔍 Advanced feature selection...")
        
        # Multiple feature selection methods
        selector_mutual = SelectKBest(score_func=mutual_info_regression, k=min(25, len(available_features)))
        selector_f = SelectKBest(score_func=f_regression, k=min(25, len(available_features)))
        
        # Combine selections
        X_train_mutual = selector_mutual.fit_transform(X_train, y_train_reg)
        X_train_f = selector_f.fit_transform(X_train, y_train_reg)
        
        # Get selected features
        mutual_features = [available_features[i] for i in selector_mutual.get_support(indices=True)]
        f_features = [available_features[i] for i in selector_f.get_support(indices=True)]
        
        # Combine and deduplicate
        selected_features = list(set(mutual_features + f_features))
        print(f"Selected {len(selected_features)} features using multiple methods")
        
        # Scale features
        scaler = PowerTransformer(method='yeo-johnson')
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        X_test_scaled = scaler.transform(X_test[selected_features])
        
        # Build sophisticated ensemble models
        print("🏗️ Building sophisticated ensemble models...")
        
        # Regression models
        reg_models = {
            'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=2, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, max_depth=12, random_state=42),
            'Ridge': Ridge(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'Bayesian Ridge': BayesianRidge(),
            'SVR': SVR(kernel='rbf', C=10.0, gamma='scale'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(300, 150, 75), max_iter=3000, learning_rate='adaptive', random_state=42)
        }
        
        # Classification models
        clf_models = {
            'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=2, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=12, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=3000, C=10.0),
            'SVC': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(300, 150, 75), max_iter=3000, learning_rate='adaptive', random_state=42)
        }
        
        # Train individual models
        reg_results = {}
        clf_results = {}
        
        for name, model in reg_models.items():
            print(f"   Training {name} regression...")
            try:
                model.fit(X_train_scaled, y_train_reg)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test_reg, y_pred)
                mae = mean_absolute_error(y_test_reg, y_pred)
                r2 = r2_score(y_test_reg, y_pred)
                
                reg_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
                
                print(f"      MAE: {mae:.2f}, R²: {r2:.3f}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        for name, model in clf_models.items():
            print(f"   Training {name} classification...")
            try:
                model.fit(X_train_scaled, y_train_clf)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test_clf, y_pred)
                
                clf_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'accuracy': accuracy
                }
                
                print(f"      Accuracy: {accuracy:.3f}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Build sophisticated stacking ensemble
        print("🗳️ Building sophisticated stacking ensemble...")
        
        # Stacking regressor
        stacking_reg = StackingRegressor(
            estimators=[
                ('rf', reg_results['Random Forest']['model']),
                ('gb', reg_results['Gradient Boosting']['model']),
                ('ridge', reg_results['Ridge']['model']),
                ('elastic', reg_results['ElasticNet']['model']),
                ('bayesian', reg_results['Bayesian Ridge']['model'])
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=5
        )
        
        stacking_reg.fit(X_train_scaled, y_train_reg)
        stacking_reg_pred = stacking_reg.predict(X_test_scaled)
        
        stacking_reg_mse = mean_squared_error(y_test_reg, stacking_reg_pred)
        stacking_reg_mae = mean_absolute_error(y_test_reg, stacking_reg_pred)
        stacking_reg_r2 = r2_score(y_test_reg, stacking_reg_pred)
        
        print(f"   Sophisticated Stacking Regression:")
        print(f"      MAE: {stacking_reg_mae:.2f}, R²: {stacking_reg_r2:.3f}")
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', clf_results['Random Forest']['model']),
                ('gb', clf_results['Gradient Boosting']['model']),
                ('lr', clf_results['Logistic Regression']['model']),
                ('svc', clf_results['SVC']['model'])
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        stacking_clf.fit(X_train_scaled, y_train_clf)
        stacking_clf_pred = stacking_clf.predict(X_test_scaled)
        stacking_clf_prob = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        
        stacking_clf_accuracy = accuracy_score(y_test_clf, stacking_clf_pred)
        
        print(f"   Sophisticated Stacking Classification:")
        print(f"      Accuracy: {stacking_clf_accuracy:.3f}")
        
        # Store models and results
        self.models = {
            'regression': reg_results,
            'classification': clf_results,
            'stacking_regression': stacking_reg,
            'stacking_classification': stacking_clf
        }
        
        self.scalers = {
            'scaler': scaler,
            'selected_features': selected_features
        }
        
        self.performance_metrics = {
            'regression': {
                'stacking_mae': stacking_reg_mae,
                'stacking_r2': stacking_reg_r2
            },
            'classification': {
                'stacking_accuracy': stacking_clf_accuracy
            }
        }
        
        return True
    
    def create_sophisticated_betting_strategy(self):
        """Create sophisticated betting strategy based on all analysis."""
        print("💰 Creating sophisticated betting strategy...")
        
        # Load betting results for strategy optimization
        profitable_bets = self.betting_results[self.betting_results['bet_amount'] > 0]
        
        # Analyze optimal parameters
        edge_ranges = [
            (0, 3, "Avoid"),
            (3, 5, "Low Confidence"),
            (5, 10, "Medium Confidence"),
            (10, 20, "High Confidence"),
            (20, 100, "Very High Confidence")
        ]
        
        strategy_params = {}
        
        for min_edge, max_edge, confidence in edge_ranges:
            edge_bets = profitable_bets[(profitable_bets['edge'].abs() >= min_edge) & (profitable_bets['edge'].abs() < max_edge)]
            if len(edge_bets) > 0:
                win_rate = len(edge_bets[edge_bets['actual_profit'] > 0]) / len(edge_bets)
                avg_profit = edge_bets['actual_profit'].mean()
                roi = (edge_bets['actual_profit'].sum() / (len(edge_bets) * 100)) * 100
                
                strategy_params[confidence] = {
                    'min_edge': min_edge,
                    'max_edge': max_edge,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'roi': roi,
                    'bet_count': len(edge_bets)
                }
        
        # Create sophisticated betting rules
        self.betting_strategy = {
            'parameters': strategy_params,
            'rules': {
                'minimum_edge': 5.0,  # Based on analysis
                'optimal_edge_range': (10, 20),  # Sweet spot
                'maximum_bet_size': 100,
                'bankroll_percentage': 0.05,
                'confidence_thresholds': {
                    'high': 0.8,
                    'medium': 0.6,
                    'low': 0.4
                }
            },
            'opponent_strategy': {
                'target_opponents': ['Vanderbilt', 'Akron', 'UTEP', 'Chattanooga'],
                'avoid_opponents': ['Florida', 'Clemson', 'Arkansas'],
                'neutral_opponents': ['Kentucky', 'South Carolina', 'Missouri']
            },
            'seasonal_adjustments': {
                'early_season': 0.8,  # Reduce confidence
                'mid_season': 1.0,    # Normal confidence
                'late_season': 1.2    # Increase confidence
            }
        }
        
        print("✅ Sophisticated betting strategy created")
        return self.betting_strategy
    
    def generate_sophisticated_predictions(self):
        """Generate sophisticated predictions for future games."""
        print("🔮 Generating sophisticated predictions...")
        
        if not self.models or not self.scalers:
            print("❌ Models not trained yet")
            return None
        
        # Use stacking models for predictions
        stacking_reg = self.models['stacking_regression']
        stacking_clf = self.models['stacking_classification']
        
        # Prepare test data
        test_df = self.advanced_df[self.advanced_df['season'] == 2024].copy()
        selected_features = self.scalers['selected_features']
        scaler = self.scalers['scaler']
        
        X_test = test_df[selected_features].fillna(0)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        reg_predictions = stacking_reg.predict(X_test_scaled)
        clf_predictions = stacking_clf.predict(X_test_scaled)
        clf_probabilities = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        
        # Create sophisticated predictions
        sophisticated_predictions = []
        
        for i, (_, game) in enumerate(test_df.iterrows()):
            opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
            
            pred_diff = reg_predictions[i]
            pred_win = clf_predictions[i]
            pred_prob = clf_probabilities[i]
            
            # Calculate sophisticated edge
            default_line = 7.5
            sophisticated_edge = pred_diff - default_line
            
            # Determine betting recommendation
            recommendation = self._get_sophisticated_recommendation(
                sophisticated_edge, pred_prob, opponent, game
            )
            
            sophisticated_predictions.append({
                'game_id': game['id'],
                'opponent': opponent,
                'week': game['week'],
                'season': game['season'],
                'predicted_differential': pred_diff,
                'predicted_win_probability': pred_prob,
                'sophisticated_edge': sophisticated_edge,
                'recommendation': recommendation['action'],
                'confidence': recommendation['confidence'],
                'bet_amount': recommendation['bet_amount'],
                'expected_profit': recommendation['expected_profit']
            })
        
        return sophisticated_predictions
    
    def _get_sophisticated_recommendation(self, edge, win_prob, opponent, game):
        """Get sophisticated betting recommendation based on all analysis."""
        
        # Base recommendation
        if abs(edge) < self.betting_strategy['rules']['minimum_edge']:
            return {
                'action': 'NO BET',
                'confidence': 'Low',
                'bet_amount': 0,
                'expected_profit': 0
            }
        
        # Opponent adjustment
        opponent_multiplier = 1.0
        if opponent in self.betting_strategy['opponent_strategy']['target_opponents']:
            opponent_multiplier = 1.2
        elif opponent in self.betting_strategy['opponent_strategy']['avoid_opponents']:
            opponent_multiplier = 0.5
        
        # Season adjustment
        week = game['week']
        if week <= 3:
            season_multiplier = self.betting_strategy['seasonal_adjustments']['early_season']
        elif week >= 10:
            season_multiplier = self.betting_strategy['seasonal_adjustments']['late_season']
        else:
            season_multiplier = self.betting_strategy['seasonal_adjustments']['mid_season']
        
        # Calculate adjusted edge
        adjusted_edge = edge * opponent_multiplier * season_multiplier
        
        # Determine action
        if adjusted_edge > 0:
            action = 'BET TENNESSEE'
            bet_prob = win_prob
        else:
            action = 'BET OPPONENT'
            bet_prob = 1 - win_prob
        
        # Calculate confidence
        if abs(adjusted_edge) >= 15:
            confidence = 'Very High'
        elif abs(adjusted_edge) >= 10:
            confidence = 'High'
        elif abs(adjusted_edge) >= 5:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Calculate bet amount
        base_amount = self.betting_strategy['rules']['maximum_bet_size']
        confidence_multiplier = {
            'Very High': 1.0,
            'High': 0.8,
            'Medium': 0.6,
            'Low': 0.4
        }
        
        bet_amount = base_amount * confidence_multiplier[confidence]
        
        # Calculate expected profit
        expected_profit = bet_amount * (bet_prob - 0.5) * 2
        
        return {
            'action': action,
            'confidence': confidence,
            'bet_amount': bet_amount,
            'expected_profit': expected_profit
        }
    
    def save_sophisticated_analysis(self):
        """Save sophisticated analysis results."""
        print("💾 Saving sophisticated analysis...")
        
        # Save predictions
        predictions = self.generate_sophisticated_predictions()
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            predictions_df.to_csv('analysis/results/sophisticated_predictions.csv', index=False)
            print("✅ Sophisticated predictions saved")
        
        # Save strategy
        strategy_df = pd.DataFrame([self.betting_strategy['rules']])
        strategy_df.to_csv('analysis/results/sophisticated_strategy.csv', index=False)
        print("✅ Sophisticated strategy saved")
        
        # Save performance metrics
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv('analysis/results/sophisticated_performance.csv', index=False)
        print("✅ Sophisticated performance metrics saved")
        
        return True

def main():
    print("🚀 Sophisticated CFB Betting Model")
    print("=" * 50)
    
    # Initialize model
    model = SophisticatedCFBModel()
    
    # Load organized data
    if not model.load_organized_data():
        return
    
    # Create advanced features
    model.create_advanced_features()
    
    # Build sophisticated models
    if not model.build_sophisticated_models():
        return
    
    # Create sophisticated betting strategy
    model.create_sophisticated_betting_strategy()
    
    # Generate predictions
    predictions = model.generate_sophisticated_predictions()
    
    # Save analysis
    model.save_sophisticated_analysis()
    
    # Display results
    print(f"\n🎯 Sophisticated Model Results:")
    print("-" * 30)
    
    if predictions:
        total_bets = len([p for p in predictions if p['bet_amount'] > 0])
        total_expected_profit = sum(p['expected_profit'] for p in predictions)
        
        print(f"📊 Total Betting Opportunities: {len(predictions)}")
        print(f"💰 Recommended Bets: {total_bets}")
        print(f"📈 Expected Total Profit: ${total_expected_profit:+.2f}")
        print(f"🎯 Average Expected Profit: ${total_expected_profit/total_bets:+.2f}" if total_bets > 0 else "N/A")
        
        print(f"\n🏆 Top Recommendations:")
        for i, pred in enumerate(predictions[:5]):
            if pred['bet_amount'] > 0:
                print(f"   {i+1}. Week {pred['week']}: {pred['recommendation']} vs {pred['opponent']}")
                print(f"      Edge: {pred['sophisticated_edge']:+.1f} | Confidence: {pred['confidence']}")
                print(f"      Bet: ${pred['bet_amount']:.0f} | Expected: ${pred['expected_profit']:+.2f}")
    
    print(f"\n✅ Sophisticated model analysis complete!")
    print(f"📁 Results saved in analysis/results/")

if __name__ == "__main__":
    main()
