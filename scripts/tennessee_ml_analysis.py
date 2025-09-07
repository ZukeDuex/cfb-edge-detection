#!/usr/bin/env python3
"""Fetch and analyze Tennessee stats data for machine learning predictions."""

import requests
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🤖 Tennessee Stats Analysis & ML Prediction Model")
    print("=" * 60)
    
    # Load existing data
    try:
        games_df = pd.read_csv('tennessee_games_2022_2024.csv')
        odds_df = pd.read_csv('tennessee_odds_2022_2024.csv')
        print(f"✅ Loaded {len(games_df)} games and {len(odds_df)} odds entries")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize API settings
    api_key = settings.cfbd_api_key
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    print(f"🔑 Using CFBD API Key: {api_key[:10]}...")
    
    # Fetch stats for each game
    print(f"\n📊 Fetching game stats...")
    all_stats = []
    
    for _, game in games_df.iterrows():
        game_id = game['id']
        season = game['season']
        week = game['week']
        home_team = game['homeTeam']
        away_team = game['awayTeam']
        
        print(f"   🏈 {season} Week {week}: {away_team} @ {home_team}")
        
        try:
            # Fetch team stats for this game
            stats_data = fetch_game_stats(api_key, headers, game_id, season, week)
            
            if stats_data:
                all_stats.extend(stats_data)
                print(f"      ✅ Found stats data")
            else:
                print(f"      ⚠️  No stats data found")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    if not all_stats:
        print("❌ No stats data found. Creating synthetic features for analysis...")
        # Create synthetic features based on game data
        all_stats = create_synthetic_features(games_df)
    
    # Create features DataFrame
    stats_df = pd.DataFrame(all_stats)
    
    # Merge with games data
    print(f"\n🔗 Merging stats with game results...")
    merged_df = merge_stats_with_games(stats_df, games_df, odds_df)
    
    # Feature engineering
    print(f"\n⚙️  Engineering features...")
    features_df = engineer_features(merged_df)
    
    # Machine learning analysis
    print(f"\n🤖 Running machine learning analysis...")
    run_ml_analysis(features_df)
    
    # Save results
    filename = 'tennessee_ml_analysis.csv'
    features_df.to_csv(filename, index=False)
    print(f"💾 Complete analysis saved to: {filename}")

def fetch_game_stats(api_key, headers, game_id, season, week):
    """Fetch stats for a specific game."""
    stats_data = []
    
    try:
        # Fetch team stats
        url = f'https://api.collegefootballdata.com/games/teams'
        params = {'gameId': game_id}
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            game_stats = response.json()
            
            for team_stats in game_stats:
                team_name = team_stats.get('school', '')
                stats = team_stats.get('stats', [])
                
                # Extract key stats
                stat_dict = {
                    'game_id': game_id,
                    'season': season,
                    'week': week,
                    'team': team_name,
                    'is_tennessee': team_name == 'Tennessee'
                }
                
                # Parse stats
                for stat in stats:
                    category = stat.get('category', '')
                    stat_value = stat.get('stat', 0)
                    
                    if category == 'rushing':
                        stat_dict['rushing_yards'] = stat_value
                    elif category == 'passing':
                        stat_dict['passing_yards'] = stat_value
                    elif category == 'totalYards':
                        stat_dict['total_yards'] = stat_value
                    elif category == 'turnovers':
                        stat_dict['turnovers'] = stat_value
                    elif category == 'firstDowns':
                        stat_dict['first_downs'] = stat_value
                    elif category == 'penalties':
                        stat_dict['penalties'] = stat_value
                    elif category == 'possessionTime':
                        stat_dict['possession_time'] = stat_value
                
                stats_data.append(stat_dict)
                
    except Exception as e:
        print(f"      Error fetching stats: {e}")
    
    return stats_data

def create_synthetic_features(games_df):
    """Create synthetic features when stats aren't available."""
    synthetic_stats = []
    
    for _, game in games_df.iterrows():
        # Create synthetic stats based on game outcomes
        home_points = game['homePoints'] if pd.notna(game['homePoints']) else 0
        away_points = game['awayPoints'] if pd.notna(game['awayPoints']) else 0
        
        # Synthetic features based on scoring patterns
        synthetic_stats.append({
            'game_id': game['id'],
            'season': game['season'],
            'week': game['week'],
            'team': game['homeTeam'],
            'is_tennessee': game['homeTeam'] == 'Tennessee',
            'points_scored': home_points,
            'points_allowed': away_points,
            'point_differential': home_points - away_points,
            'total_points': home_points + away_points,
            'rushing_yards': np.random.normal(150, 50),  # Synthetic
            'passing_yards': np.random.normal(250, 75),  # Synthetic
            'total_yards': np.random.normal(400, 100),   # Synthetic
            'turnovers': np.random.poisson(1.5),         # Synthetic
            'first_downs': np.random.normal(20, 5),       # Synthetic
            'penalties': np.random.normal(6, 2),          # Synthetic
            'possession_time': np.random.normal(30, 5)    # Synthetic
        })
        
        synthetic_stats.append({
            'game_id': game['id'],
            'season': game['season'],
            'week': game['week'],
            'team': game['awayTeam'],
            'is_tennessee': game['awayTeam'] == 'Tennessee',
            'points_scored': away_points,
            'points_allowed': home_points,
            'point_differential': away_points - home_points,
            'total_points': home_points + away_points,
            'rushing_yards': np.random.normal(150, 50),  # Synthetic
            'passing_yards': np.random.normal(250, 75),  # Synthetic
            'total_yards': np.random.normal(400, 100),   # Synthetic
            'turnovers': np.random.poisson(1.5),         # Synthetic
            'first_downs': np.random.normal(20, 5),       # Synthetic
            'penalties': np.random.normal(6, 2),          # Synthetic
            'possession_time': np.random.normal(30, 5)    # Synthetic
        })
    
    return synthetic_stats

def merge_stats_with_games(stats_df, games_df, odds_df):
    """Merge stats data with game results and odds."""
    merged_data = []
    
    for _, game in games_df.iterrows():
        game_id = game['id']
        
        # Get stats for this game
        game_stats = stats_df[stats_df['game_id'] == game_id]
        tennessee_stats = game_stats[game_stats['is_tennessee'] == True]
        opponent_stats = game_stats[game_stats['is_tennessee'] == False]
        
        if len(tennessee_stats) > 0 and len(opponent_stats) > 0:
            tn_stats = tennessee_stats.iloc[0]
            opp_stats = opponent_stats.iloc[0]
            
            # Get odds for this game
            game_odds = odds_df[odds_df['game_id'] == game_id]
            
            # Create merged record
            merged_record = {
                'game_id': game_id,
                'season': game['season'],
                'week': game['week'],
                'home_team': game['homeTeam'],
                'away_team': game['awayTeam'],
                'home_points': game['homePoints'],
                'away_points': game['awayPoints'],
                'completed': game['completed'],
                'venue': game['venue'],
                
                # Tennessee stats
                'tn_points_scored': tn_stats['points_scored'],
                'tn_points_allowed': tn_stats['points_allowed'],
                'tn_point_differential': tn_stats['point_differential'],
                'tn_rushing_yards': tn_stats['rushing_yards'],
                'tn_passing_yards': tn_stats['passing_yards'],
                'tn_total_yards': tn_stats['total_yards'],
                'tn_turnovers': tn_stats['turnovers'],
                'tn_first_downs': tn_stats['first_downs'],
                'tn_penalties': tn_stats['penalties'],
                'tn_possession_time': tn_stats['possession_time'],
                
                # Opponent stats
                'opp_points_scored': opp_stats['points_scored'],
                'opp_points_allowed': opp_stats['points_allowed'],
                'opp_point_differential': opp_stats['point_differential'],
                'opp_rushing_yards': opp_stats['rushing_yards'],
                'opp_passing_yards': opp_stats['passing_yards'],
                'opp_total_yards': opp_stats['total_yards'],
                'opp_turnovers': opp_stats['turnovers'],
                'opp_first_downs': opp_stats['first_downs'],
                'opp_penalties': opp_stats['penalties'],
                'opp_possession_time': opp_stats['possession_time'],
                
                # Betting data
                'spread': None,
                'moneyline': None,
                'total_points': None
            }
            
            # Add betting data if available
            if not game_odds.empty:
                spreads = game_odds[game_odds['market'] == 'spreads']
                moneylines = game_odds[game_odds['market'] == 'h2h']
                totals = game_odds[game_odds['market'] == 'totals']
                
                if not spreads.empty:
                    tn_spread = spreads[spreads['outcome'].str.contains('Tennessee', case=False, na=False)]
                    if not tn_spread.empty:
                        merged_record['spread'] = tn_spread.iloc[0]['point']
                
                if not moneylines.empty:
                    tn_ml = moneylines[moneylines['outcome'].str.contains('Tennessee', case=False, na=False)]
                    if not tn_ml.empty:
                        merged_record['moneyline'] = tn_ml.iloc[0]['price']
                
                if not totals.empty:
                    over_total = totals[totals['outcome'].str.contains('Over', case=False, na=False)]
                    if not over_total.empty:
                        merged_record['total_points'] = over_total.iloc[0]['point']
            
            merged_data.append(merged_record)
    
    return pd.DataFrame(merged_data)

def engineer_features(df):
    """Create additional features for machine learning."""
    features_df = df.copy()
    
    # Calculate derived features
    features_df['tn_yards_per_point'] = features_df['tn_total_yards'] / (features_df['tn_points_scored'] + 1)
    features_df['opp_yards_per_point'] = features_df['opp_total_yards'] / (features_df['opp_points_scored'] + 1)
    
    features_df['tn_turnover_margin'] = features_df['opp_turnovers'] - features_df['tn_turnovers']
    features_df['tn_yard_margin'] = features_df['tn_total_yards'] - features_df['opp_total_yards']
    features_df['tn_time_of_possession_margin'] = features_df['tn_possession_time'] - features_df['opp_possession_time']
    
    # Efficiency metrics
    features_df['tn_offensive_efficiency'] = features_df['tn_points_scored'] / (features_df['tn_total_yards'] + 1)
    features_df['tn_defensive_efficiency'] = features_df['opp_points_scored'] / (features_df['opp_total_yards'] + 1)
    
    # Game context features
    features_df['is_home_game'] = features_df['home_team'] == 'Tennessee'
    features_df['is_conference_game'] = features_df['venue'].str.contains('Stadium', na=False)
    
    # Create target variables
    features_df['tennessee_won'] = False
    features_df['tennessee_covered'] = False
    
    for idx, row in features_df.iterrows():
        if row['completed'] and pd.notna(row['home_points']) and pd.notna(row['away_points']):
            # Determine if Tennessee won
            if row['home_team'] == 'Tennessee':
                tennessee_won = row['home_points'] > row['away_points']
                tennessee_score = row['home_points']
                opponent_score = row['away_points']
            else:
                tennessee_won = row['away_points'] > row['home_points']
                tennessee_score = row['away_points']
                opponent_score = row['home_points']
            
            features_df.at[idx, 'tennessee_won'] = tennessee_won
            
            # Check if Tennessee covered the spread
            if pd.notna(row['spread']):
                spread = row['spread']
                if row['home_team'] == 'Tennessee':
                    covered = (tennessee_score - opponent_score) > spread
                else:
                    covered = (opponent_score - tennessee_score) < spread
                
                features_df.at[idx, 'tennessee_covered'] = covered
    
    return features_df

def run_ml_analysis(df):
    """Run machine learning analysis to find predictive indicators."""
    
    # Prepare features
    feature_columns = [
        'tn_point_differential', 'tn_yard_margin', 'tn_turnover_margin',
        'tn_time_of_possession_margin', 'tn_offensive_efficiency', 'tn_defensive_efficiency',
        'tn_yards_per_point', 'opp_yards_per_point', 'is_home_game', 'is_conference_game',
        'spread', 'moneyline', 'total_points'
    ]
    
    # Remove rows with missing target data
    df_clean = df.dropna(subset=['tennessee_won'])
    
    if len(df_clean) < 10:
        print("❌ Not enough data for machine learning analysis")
        return
    
    # Prepare features and targets
    X = df_clean[feature_columns].fillna(0)
    y_win = df_clean['tennessee_won']
    y_cover = df_clean['tennessee_covered'].fillna(False)
    
    # Split data
    X_train, X_test, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.3, random_state=42)
    _, _, y_cover_train, y_cover_test = train_test_split(X, y_cover, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training data: {len(X_train)} games, Testing data: {len(X_test)} games")
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    # Test win prediction
    print(f"\n🏆 Win Prediction Analysis:")
    print("-" * 40)
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_win_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_win_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_win_test, y_pred)
        print(f"{name}: {accuracy:.3f} accuracy")
        
        if name == 'Random Forest':
            # Show feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top Predictive Features for Wins:")
            for _, row in feature_importance.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Test spread coverage prediction
    print(f"\n📏 Spread Coverage Prediction Analysis:")
    print("-" * 40)
    
    # Only use games with spread data
    spread_mask = df_clean['tennessee_covered'].notna()
    X_spread = X[spread_mask]
    y_spread = y_cover[spread_mask]
    
    if len(X_spread) > 5:
        X_train_spread, X_test_spread, y_train_spread, y_test_spread = train_test_split(
            X_spread, y_spread, test_size=0.3, random_state=42
        )
        
        X_train_spread_scaled = scaler.fit_transform(X_train_spread)
        X_test_spread_scaled = scaler.transform(X_test_spread)
        
        for name, model in models.items():
            if name == 'Logistic Regression':
                model.fit(X_train_spread_scaled, y_train_spread)
                y_pred_spread = model.predict(X_test_spread_scaled)
            else:
                model.fit(X_train_spread, y_train_spread)
                y_pred_spread = model.predict(X_test_spread)
            
            accuracy = accuracy_score(y_test_spread, y_pred_spread)
            print(f"{name}: {accuracy:.3f} accuracy")
    
    # Statistical analysis
    print(f"\n📈 Statistical Analysis:")
    print("-" * 40)
    
    # Analyze correlations
    numeric_features = df_clean.select_dtypes(include=[np.number])
    correlations = numeric_features.corr()['tennessee_won'].abs().sort_values(ascending=False)
    
    print(f"🔗 Top Correlations with Tennessee Wins:")
    for feature, corr in correlations.head(8).items():
        if feature != 'tennessee_won':
            print(f"   {feature}: {corr:.3f}")
    
    # Performance by feature ranges
    print(f"\n📊 Performance Analysis:")
    print("-" * 40)
    
    # Analyze by yard margin
    df_clean['yard_margin_category'] = pd.cut(df_clean['tn_yard_margin'], 
                                            bins=[-float('inf'), -50, 0, 50, float('inf')], 
                                            labels=['Large Deficit', 'Small Deficit', 'Small Advantage', 'Large Advantage'])
    
    yard_performance = df_clean.groupby('yard_margin_category')['tennessee_won'].agg(['count', 'sum', 'mean'])
    print(f"Win Rate by Yard Margin:")
    for category, stats in yard_performance.iterrows():
        if stats['count'] > 0:
            print(f"   {category}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")

if __name__ == "__main__":
    main()
