#!/usr/bin/env python3
"""
Simplified Sophisticated CFB Model - Works with Organized Analysis Structure
This model demonstrates how to leverage the organized analysis for advanced betting.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression, BayesianRidge, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import warnings
warnings.filterwarnings('ignore')

class OrganizedCFBModel:
    """
    CFB Betting Model that leverages the organized analysis structure.
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
            self.games_df = pd.read_csv('data/tennessee_games_2022_2024.csv')
            self.odds_df = pd.read_csv('data/tennessee_odds_2022_2024.csv')
            self.enhanced_df = pd.read_csv('data/tennessee_games_error_minimized.csv')
            self.betting_df = pd.read_csv('data/tennessee_games_profitable_betting.csv')
            
            # Load results
            self.betting_results = pd.read_csv('results/simple_betting_analysis.csv')
            self.strategy_summary = pd.read_csv('results/betting_strategy_summary.csv')
            
            print(f"✅ Loaded {len(self.games_df)} games, {len(self.odds_df)} betting lines")
            print(f"✅ Enhanced features: {len(self.enhanced_df.columns)} columns")
            print(f"✅ Betting results: {len(self.betting_results)} opportunities")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def create_sophisticated_features(self):
        """Create sophisticated features based on analysis insights."""
        print("🔧 Creating sophisticated features...")
        
        # Start with enhanced features
        self.sophisticated_df = self.enhanced_df.copy()
        
        # 1. Betting-specific features based on analysis
        self.sophisticated_df['is_profitable_opponent'] = self.sophisticated_df['awayTeam'].isin([
            'Alabama', 'Vanderbilt', 'Georgia', 'Akron', 'UTEP'
        ])
        self.sophisticated_df['is_avoid_opponent'] = self.sophisticated_df['awayTeam'].isin([
            'Florida', 'Clemson', 'Arkansas'
        ])
        
        # 2. Edge-based features
        self.sophisticated_df['predicted_edge'] = self.sophisticated_df['tennessee_point_differential'] - 7.5
        self.sophisticated_df['edge_category'] = pd.cut(
            self.sophisticated_df['predicted_edge'].abs(),
            bins=[0, 3, 10, 20, 100],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        # 3. Opponent strength features
        opponent_strength = {
            'Alabama': 0.95, 'Georgia': 0.95, 'Ohio State': 0.90, 'Clemson': 0.90,
            'LSU': 0.85, 'Florida': 0.80, 'Auburn': 0.80, 'Texas A&M': 0.75,
            'Kentucky': 0.70, 'South Carolina': 0.70, 'Missouri': 0.65,
            'Vanderbilt': 0.30, 'Akron': 0.20, 'UTEP': 0.15, 'Chattanooga': 0.10
        }
        self.sophisticated_df['opponent_strength'] = self.sophisticated_df['awayTeam'].map(opponent_strength).fillna(0.50)
        
        # 4. Betting confidence features
        self.sophisticated_df['betting_confidence'] = np.where(
            self.sophisticated_df['predicted_edge'].abs() > 10, 'High',
            np.where(self.sophisticated_df['predicted_edge'].abs() > 5, 'Medium', 'Low')
        )
        
        # 5. Historical performance features
        self.sophisticated_df['historical_vs_opponent'] = self.sophisticated_df.groupby('awayTeam')['tennessee_point_differential'].transform('mean')
        self.sophisticated_df['historical_win_rate_vs_opponent'] = self.sophisticated_df.groupby('awayTeam')['tennessee_won'].transform('mean')
        
        print(f"✅ Created {len(self.sophisticated_df.columns) - len(self.enhanced_df.columns)} sophisticated features")
        
        return self.sophisticated_df
    
    def build_organized_models(self):
        """Build models using organized analysis data."""
        print("🤖 Building organized models...")
        
        # Prepare features
        feature_columns = [
            'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
            'homePregameElo', 'awayPregameElo', 'homePostgameWinProbability', 'awayPostgameWinProbability',
            'excitement_index', 'opponent_strength', 'is_profitable_opponent', 'is_avoid_opponent',
            'predicted_edge', 'betting_confidence', 'historical_vs_opponent', 'historical_win_rate_vs_opponent'
        ]
        
        # Add enhanced features
        enhanced_features = [col for col in self.sophisticated_df.columns if col not in feature_columns and col not in ['id', 'season', 'seasonType', 'startDate', 'homeTeam', 'awayTeam', 'homePoints', 'awayPoints', 'tennessee_won', 'tennessee_point_differential']]
        feature_columns.extend(enhanced_features)
        
        # Filter to available features
        available_features = [f for f in feature_columns if f in self.sophisticated_df.columns]
        
        print(f"📊 Using {len(available_features)} features for modeling")
        
        # Split data
        train_df = self.sophisticated_df[self.sophisticated_df['season'].isin([2022, 2023])].copy()
        test_df = self.sophisticated_df[self.sophisticated_df['season'] == 2024].copy()
        
        if len(test_df) == 0:
            print("❌ No 2024 data for testing")
            return False
        
        # Prepare data - handle categorical columns
        X_train = train_df[available_features].copy()
        X_test = test_df[available_features].copy()
        
        # Convert categorical columns to numeric
        for col in X_train.columns:
            if X_train[col].dtype == 'category':
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
            
            # Convert string columns to numeric
            if X_train[col].dtype == 'object':
                # Try to convert to numeric, if fails, use label encoding
                try:
                    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                except:
                    # Label encode string columns
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))
        
        # Fill NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        y_train_reg = train_df['tennessee_point_differential']
        y_train_clf = train_df['tennessee_won']
        y_test_reg = test_df['tennessee_point_differential']
        y_test_clf = test_df['tennessee_won']
        
        # Feature selection
        print("🔍 Feature selection...")
        selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(available_features)))
        X_train_selected = selector.fit_transform(X_train, y_train_reg)
        X_test_selected = selector.transform(X_test)
        
        selected_features = [available_features[i] for i in selector.get_support(indices=True)]
        print(f"Selected {len(selected_features)} features")
        
        # Scale features
        scaler = PowerTransformer(method='yeo-johnson')
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Build models
        print("🏗️ Building models...")
        
        # Regression models
        reg_models = {
            'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, random_state=42),
            'Ridge': Ridge(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'Bayesian Ridge': BayesianRidge(),
            'SVR': SVR(kernel='rbf', C=10.0, gamma='scale'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42)
        }
        
        # Classification models
        clf_models = {
            'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
            'SVC': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42)
        }
        
        # Train models
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
        
        # Build stacking ensemble
        print("🗳️ Building stacking ensemble...")
        
        stacking_reg = StackingRegressor(
            estimators=[
                ('rf', reg_results['Random Forest']['model']),
                ('gb', reg_results['Gradient Boosting']['model']),
                ('ridge', reg_results['Ridge']['model']),
                ('bayesian', reg_results['Bayesian Ridge']['model'])
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=5
        )
        
        stacking_reg.fit(X_train_scaled, y_train_reg)
        stacking_reg_pred = stacking_reg.predict(X_test_scaled)
        
        stacking_reg_mae = mean_absolute_error(y_test_reg, stacking_reg_pred)
        stacking_reg_r2 = r2_score(y_test_reg, stacking_reg_pred)
        
        print(f"   Stacking Regression: MAE: {stacking_reg_mae:.2f}, R²: {stacking_reg_r2:.3f}")
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', clf_results['Random Forest']['model']),
                ('gb', clf_results['Gradient Boosting']['model']),
                ('lr', clf_results['Logistic Regression']['model'])
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        
        stacking_clf.fit(X_train_scaled, y_train_clf)
        stacking_clf_pred = stacking_clf.predict(X_test_scaled)
        stacking_clf_prob = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        
        stacking_clf_accuracy = accuracy_score(y_test_clf, stacking_clf_pred)
        
        print(f"   Stacking Classification: Accuracy: {stacking_clf_accuracy:.3f}")
        
        # Store results
        self.models = {
            'regression': reg_results,
            'classification': clf_results,
            'stacking_regression': stacking_reg,
            'stacking_classification': stacking_clf
        }
        
        self.scalers = {
            'scaler': scaler,
            'selector': selector,
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
    
    def create_organized_betting_strategy(self):
        """Create betting strategy based on organized analysis."""
        print("💰 Creating organized betting strategy...")
        
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
        
        # Create betting strategy
        self.betting_strategy = {
            'parameters': strategy_params,
            'rules': {
                'minimum_edge': 5.0,
                'optimal_edge_range': (10, 20),
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
            }
        }
        
        print("✅ Organized betting strategy created")
        return self.betting_strategy
    
    def generate_organized_predictions(self):
        """Generate predictions using organized analysis."""
        print("🔮 Generating organized predictions...")
        
        if not self.models or not self.scalers:
            print("❌ Models not trained yet")
            return None
        
        # Use stacking models
        stacking_reg = self.models['stacking_regression']
        stacking_clf = self.models['stacking_classification']
        
        # Prepare test data
        test_df = self.sophisticated_df[self.sophisticated_df['season'] == 2024].copy()
        selected_features = self.scalers['selected_features']
        scaler = self.scalers['scaler']
        selector = self.scalers['selector']
        
        X_test = test_df[selected_features].fillna(0)
        X_test_selected = selector.transform(X_test)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Make predictions
        reg_predictions = stacking_reg.predict(X_test_scaled)
        clf_predictions = stacking_clf.predict(X_test_scaled)
        clf_probabilities = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        
        # Create predictions
        organized_predictions = []
        
        for i, (_, game) in enumerate(test_df.iterrows()):
            opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
            
            pred_diff = reg_predictions[i]
            pred_win = clf_predictions[i]
            pred_prob = clf_probabilities[i]
            
            # Calculate edge
            default_line = 7.5
            edge = pred_diff - default_line
            
            # Determine recommendation
            recommendation = self._get_organized_recommendation(edge, pred_prob, opponent)
            
            organized_predictions.append({
                'game_id': game['id'],
                'opponent': opponent,
                'week': game['week'],
                'season': game['season'],
                'predicted_differential': pred_diff,
                'predicted_win_probability': pred_prob,
                'edge': edge,
                'recommendation': recommendation['action'],
                'confidence': recommendation['confidence'],
                'bet_amount': recommendation['bet_amount'],
                'expected_profit': recommendation['expected_profit']
            })
        
        return organized_predictions
    
    def _get_organized_recommendation(self, edge, win_prob, opponent):
        """Get betting recommendation based on organized analysis."""
        
        # Base recommendation
        if abs(edge) < self.betting_strategy['rules']['minimum_edge']:
            return {
                'action': 'NO BET',
                'confidence': 'Low',
                'bet_amount': 0,
                'expected_profit': 0
            }
        
        # Opponent adjustment
        if opponent in self.betting_strategy['opponent_strategy']['target_opponents']:
            edge *= 1.2
        elif opponent in self.betting_strategy['opponent_strategy']['avoid_opponents']:
            edge *= 0.5
        
        # Determine action
        if edge > 0:
            action = 'BET TENNESSEE'
            bet_prob = win_prob
        else:
            action = 'BET OPPONENT'
            bet_prob = 1 - win_prob
        
        # Calculate confidence
        if abs(edge) >= 15:
            confidence = 'Very High'
        elif abs(edge) >= 10:
            confidence = 'High'
        elif abs(edge) >= 5:
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
    
    def save_organized_results(self):
        """Save organized analysis results."""
        print("💾 Saving organized results...")
        
        # Save predictions
        predictions = self.generate_organized_predictions()
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            predictions_df.to_csv('results/organized_predictions.csv', index=False)
            print("✅ Organized predictions saved")
        
        # Save strategy
        strategy_df = pd.DataFrame([self.betting_strategy['rules']])
        strategy_df.to_csv('results/organized_strategy.csv', index=False)
        print("✅ Organized strategy saved")
        
        # Save performance metrics
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv('results/organized_performance.csv', index=False)
        print("✅ Organized performance metrics saved")
        
        return True

def main():
    print("🚀 Organized CFB Betting Model")
    print("=" * 50)
    
    # Initialize model
    model = OrganizedCFBModel()
    
    # Load organized data
    if not model.load_organized_data():
        return
    
    # Create sophisticated features
    model.create_sophisticated_features()
    
    # Build models
    if not model.build_organized_models():
        return
    
    # Create betting strategy
    model.create_organized_betting_strategy()
    
    # Generate predictions
    predictions = model.generate_organized_predictions()
    
    # Save results
    model.save_organized_results()
    
    # Display results
    print(f"\n🎯 Organized Model Results:")
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
                print(f"      Edge: {pred['edge']:+.1f} | Confidence: {pred['confidence']}")
                print(f"      Bet: ${pred['bet_amount']:.0f} | Expected: ${pred['expected_profit']:+.2f}")
    
    print(f"\n✅ Organized model analysis complete!")
    print(f"📁 Results saved in results/")

if __name__ == "__main__":
    main()
