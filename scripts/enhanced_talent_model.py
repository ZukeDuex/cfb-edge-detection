#!/usr/bin/env python3
"""Enhanced ML model using CFBD talent data and synthetic stats."""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import requests
import time

def main():
    print("📊 Enhanced ML Model with CFBD Talent Data")
    print("=" * 60)
    
    # Load Tennessee games data
    try:
        tennessee_df = pd.read_csv('tennessee_games_2022_2024.csv')
        print(f"✅ Loaded {len(tennessee_df)} Tennessee games")
    except Exception as e:
        print(f"❌ Error loading Tennessee games: {e}")
        return
    
    # Initialize API settings
    api_key = settings.cfbd_api_key
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    print(f"🔑 Using CFBD API Key: {api_key[:10]}...")
    
    # Fetch team talent data (this worked successfully)
    print(f"\n🎯 Fetching team talent data...")
    talent_data = fetch_team_talent_data(api_key, headers)
    
    # Create synthetic stats based on talent tiers
    print(f"\n📊 Creating synthetic stats based on talent tiers...")
    synthetic_stats = create_talent_based_stats(tennessee_df, talent_data)
    
    # Create enhanced dataset
    print(f"\n🔗 Creating enhanced dataset...")
    enhanced_df = create_enhanced_dataset(tennessee_df, talent_data, synthetic_stats)
    
    # Analyze enhanced features
    print(f"\n📊 Analyzing enhanced features...")
    analyze_enhanced_features(enhanced_df)
    
    # Build enhanced ML model
    print(f"\n🤖 Building enhanced ML model...")
    build_enhanced_ml_model(enhanced_df)
    
    # Save enhanced data
    filename = 'tennessee_games_enhanced_talent.csv'
    enhanced_df.to_csv(filename, index=False)
    print(f"💾 Enhanced data saved to: {filename}")

def fetch_team_talent_data(api_key, headers):
    """Fetch team talent data (similar to cfbd_team_talent())."""
    
    talent_data = []
    
    try:
        # Fetch team talent for recent years
        for year in [2022, 2023, 2024]:
            print(f"   📊 Fetching {year} talent data...")
            
            url = f'https://api.collegefootballdata.com/talent'
            params = {'year': year}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for team in data:
                    talent_data.append({
                        'year': year,
                        'school': team.get('school'),
                        'talent': team.get('talent')
                    })
                print(f"      ✅ Found {len(data)} teams")
            else:
                print(f"      ❌ Error: {response.status_code}")
            
            time.sleep(0.5)  # Rate limiting
            
    except Exception as e:
        print(f"   ❌ Error fetching talent data: {e}")
    
    return pd.DataFrame(talent_data)

def create_talent_based_stats(tennessee_df, talent_data):
    """Create synthetic stats based on talent tiers."""
    
    synthetic_stats = []
    
    # Get unique opponents by season
    opponents_by_season = {}
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        season = game['season']
        
        if season not in opponents_by_season:
            opponents_by_season[season] = set()
        opponents_by_season[season].add(opponent)
    
    # Define stat ranges by talent tier
    talent_stat_ranges = {
        'Elite': {  # 900+ talent
            'offensive_yards_per_game': (450, 550),
            'defensive_yards_per_game': (280, 350),
            'offensive_points_per_game': (35, 45),
            'defensive_points_per_game': (15, 25),
            'rushing_yards_per_game': (180, 220),
            'passing_yards_per_game': (250, 350),
            'turnover_margin': (0.5, 2.0),
            'third_down_conversion': (0.45, 0.55),
            'red_zone_efficiency': (0.85, 0.95),
            'sacks_per_game': (2.5, 4.0),
            'interceptions_per_game': (1.0, 2.5)
        },
        'Strong': {  # 800-899 talent
            'offensive_yards_per_game': (400, 480),
            'defensive_yards_per_game': (320, 400),
            'offensive_points_per_game': (28, 38),
            'defensive_points_per_game': (20, 30),
            'rushing_yards_per_game': (150, 200),
            'passing_yards_per_game': (220, 320),
            'turnover_margin': (0.0, 1.5),
            'third_down_conversion': (0.40, 0.50),
            'red_zone_efficiency': (0.80, 0.90),
            'sacks_per_game': (2.0, 3.5),
            'interceptions_per_game': (0.8, 2.0)
        },
        'Average': {  # 700-799 talent
            'offensive_yards_per_game': (350, 420),
            'defensive_yards_per_game': (380, 450),
            'offensive_points_per_game': (22, 32),
            'defensive_points_per_game': (25, 35),
            'rushing_yards_per_game': (120, 180),
            'passing_yards_per_game': (200, 280),
            'turnover_margin': (-0.5, 1.0),
            'third_down_conversion': (0.35, 0.45),
            'red_zone_efficiency': (0.75, 0.85),
            'sacks_per_game': (1.5, 3.0),
            'interceptions_per_game': (0.5, 1.5)
        },
        'Weak': {  # 600-699 talent
            'offensive_yards_per_game': (300, 380),
            'defensive_yards_per_game': (420, 500),
            'offensive_points_per_game': (18, 28),
            'defensive_points_per_game': (30, 40),
            'rushing_yards_per_game': (100, 160),
            'passing_yards_per_game': (180, 260),
            'turnover_margin': (-1.0, 0.5),
            'third_down_conversion': (0.30, 0.40),
            'red_zone_efficiency': (0.70, 0.80),
            'sacks_per_game': (1.0, 2.5),
            'interceptions_per_game': (0.3, 1.2)
        },
        'FCS': {  # <600 talent
            'offensive_yards_per_game': (250, 350),
            'defensive_yards_per_game': (450, 550),
            'offensive_points_per_game': (15, 25),
            'defensive_points_per_game': (35, 45),
            'rushing_yards_per_game': (80, 140),
            'passing_yards_per_game': (160, 240),
            'turnover_margin': (-1.5, 0.0),
            'third_down_conversion': (0.25, 0.35),
            'red_zone_efficiency': (0.65, 0.75),
            'sacks_per_game': (0.5, 2.0),
            'interceptions_per_game': (0.2, 1.0)
        }
    }
    
    for season, opponents in opponents_by_season.items():
        for opponent in opponents:
            # Get opponent talent
            opponent_talent = talent_data[
                (talent_data['school'] == opponent) & 
                (talent_data['year'] == season)
            ]
            
            if len(opponent_talent) > 0:
                talent_score = opponent_talent['talent'].iloc[0]
            else:
                # Default talent based on known team strength
                default_talents = {
                    'Alabama': 950, 'Georgia': 960, 'Ohio State': 940, 'Clemson': 920,
                    'Florida': 820, 'LSU': 850, 'Oklahoma': 880, 'Iowa': 800, 'Texas A&M': 830,
                    'Kentucky': 750, 'South Carolina': 720, 'Missouri': 740, 'Arkansas': 730,
                    'NC State': 760, 'UTSA': 680, 'Mississippi State': 750,
                    'Vanderbilt': 650, 'Ball State': 600, 'Akron': 580, 'Virginia': 620,
                    'Kent State': 590, 'UTEP': 570, 'UConn': 550,
                    'UT Martin': 400, 'Austin Peay': 350, 'Chattanooga': 380
                }
                talent_score = default_talents.get(opponent, 700)
            
            # Determine talent tier
            if talent_score >= 900:
                tier = 'Elite'
            elif talent_score >= 800:
                tier = 'Strong'
            elif talent_score >= 700:
                tier = 'Average'
            elif talent_score >= 600:
                tier = 'Weak'
            else:
                tier = 'FCS'
            
            # Generate stats based on tier
            stat_ranges = talent_stat_ranges[tier]
            stats = {
                'season': season,
                'team': opponent,
                'talent_score': talent_score,
                'talent_tier': tier
            }
            
            # Generate stats within realistic ranges
            for stat_name, (min_val, max_val) in stat_ranges.items():
                stats[stat_name] = np.random.uniform(min_val, max_val)
            
            # Add some additional stats
            stats['time_of_possession'] = np.random.uniform(28, 32)  # minutes
            stats['penalties_per_game'] = np.random.uniform(6, 10)
            stats['penalty_yards_per_game'] = np.random.uniform(50, 80)
            stats['fumbles_per_game'] = np.random.uniform(0.5, 1.5)
            
            synthetic_stats.append(stats)
    
    return pd.DataFrame(synthetic_stats)

def create_enhanced_dataset(tennessee_df, talent_data, stats_data):
    """Create enhanced dataset with talent and stats data."""
    
    enhanced_data = []
    
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        season = game['season']
        
        # Get opponent talent
        opponent_talent = talent_data[
            (talent_data['school'] == opponent) & 
            (talent_data['year'] == season)
        ]
        
        if len(opponent_talent) > 0:
            talent_score = opponent_talent['talent'].iloc[0]
        else:
            # Default talent
            default_talents = {
                'Alabama': 950, 'Georgia': 960, 'Ohio State': 940, 'Clemson': 920,
                'Florida': 820, 'LSU': 850, 'Oklahoma': 880, 'Iowa': 800, 'Texas A&M': 830,
                'Kentucky': 750, 'South Carolina': 720, 'Missouri': 740, 'Arkansas': 730,
                'NC State': 760, 'UTSA': 680, 'Mississippi State': 750,
                'Vanderbilt': 650, 'Ball State': 600, 'Akron': 580, 'Virginia': 620,
                'Kent State': 590, 'UTEP': 570, 'UConn': 550,
                'UT Martin': 400, 'Austin Peay': 350, 'Chattanooga': 380
            }
            talent_score = default_talents.get(opponent, 700)
        
        # Get opponent stats
        opponent_stats = stats_data[
            (stats_data['team'] == opponent) & 
            (stats_data['season'] == season)
        ]
        
        # Create enhanced record
        enhanced_record = game.to_dict()
        
        # Add talent data
        enhanced_record['opponent_talent'] = talent_score
        enhanced_record['opponent_talent_tier'] = get_talent_tier(talent_score)
        enhanced_record['talent_difference'] = get_tennessee_talent(season) - talent_score
        
        # Add stats data
        if len(opponent_stats) > 0:
            stats = opponent_stats.iloc[0]
            for key, value in stats.items():
                if key not in ['season', 'team', 'talent_score', 'talent_tier'] and pd.notna(value):
                    enhanced_record[f'opponent_{key}'] = value
        
        # Calculate Tennessee performance
        tennessee_home = game['homeTeam'] == 'Tennessee'
        tennessee_points = game['homePoints'] if tennessee_home else game['awayPoints']
        opponent_points = game['awayPoints'] if tennessee_home else game['homePoints']
        tennessee_won = tennessee_points > opponent_points
        tennessee_point_differential = tennessee_points - opponent_points
        
        enhanced_record['tennessee_won'] = tennessee_won
        enhanced_record['tennessee_point_differential'] = tennessee_point_differential
        enhanced_record['is_home_game'] = tennessee_home
        
        enhanced_data.append(enhanced_record)
    
    return pd.DataFrame(enhanced_data)

def get_talent_tier(talent_score):
    """Convert talent score to tier."""
    if talent_score >= 900:
        return 'Elite'
    elif talent_score >= 800:
        return 'Strong'
    elif talent_score >= 700:
        return 'Average'
    elif talent_score >= 600:
        return 'Weak'
    else:
        return 'FCS'

def get_tennessee_talent(season):
    """Get Tennessee talent score for a given season."""
    # Approximate Tennessee talent scores
    tennessee_talent = {
        2022: 850,
        2023: 860,
        2024: 870
    }
    return tennessee_talent.get(season, 850)

def analyze_enhanced_features(enhanced_df):
    """Analyze the enhanced features."""
    
    print(f"\n📊 Enhanced Features Analysis:")
    print("-" * 50)
    
    # Analyze by talent tier
    print(f"🎯 Tennessee Performance by Opponent Talent Tier:")
    
    talent_performance = enhanced_df.groupby('opponent_talent_tier').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean',
        'opponent_talent': 'mean'
    })
    
    for tier, stats in talent_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        avg_talent = stats[('opponent_talent', 'mean')]
        
        print(f"   vs {tier} talent: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f} | Avg talent: {avg_talent:.0f}")
    
    # Analyze by talent difference
    print(f"\n📈 Tennessee Performance by Talent Difference:")
    
    enhanced_df['talent_difference_tier'] = pd.cut(
        enhanced_df['talent_difference'], 
        bins=[-200, -50, 50, 200], 
        labels=['Underdog', 'Even', 'Favorite']
    )
    
    talent_diff_performance = enhanced_df.groupby('talent_difference_tier').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    for tier, stats in talent_diff_performance.iterrows():
        if pd.notna(tier):
            wins = stats[('tennessee_won', 'sum')]
            total = stats[('tennessee_won', 'count')]
            win_pct = stats[('tennessee_won', 'mean')]
            avg_diff = stats[('tennessee_point_differential', 'mean')]
            
            print(f"   {tier}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")
    
    # Analyze by home/away
    print(f"\n🏠 Tennessee Performance by Location:")
    
    location_performance = enhanced_df.groupby('is_home_game').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    for location, stats in location_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        
        location_name = "Home" if location else "Away"
        print(f"   {location_name}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")

def build_enhanced_ml_model(enhanced_df):
    """Build enhanced ML model with talent and stats data."""
    
    print(f"\n🤖 Building Enhanced ML Model:")
    print("-" * 50)
    
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    
    # Prepare enhanced features
    base_features = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    # Add talent features
    talent_features = ['opponent_talent', 'talent_difference']
    
    # Add available stats features
    stat_features = [col for col in enhanced_df.columns if col.startswith('opponent_') and col not in ['opponent_talent', 'opponent_talent_tier']]
    
    enhanced_features = base_features + talent_features + stat_features
    
    # Filter to available features
    available_features = [f for f in enhanced_features if f in enhanced_df.columns]
    
    print(f"📊 Using {len(available_features)} features for ML")
    
    # Split data
    train_df = enhanced_df[enhanced_df['season'].isin([2022, 2023])].copy()
    test_df = enhanced_df[enhanced_df['season'] == 2024].copy()
    
    if len(test_df) == 0:
        print("❌ No 2024 data for testing")
        return
    
    # Prepare data
    X_train = train_df[available_features].fillna(0)
    y_train_reg = train_df['tennessee_point_differential']
    y_train_clf = train_df['tennessee_won']
    
    X_test = test_df[available_features].fillna(0)
    y_test_reg = test_df['tennessee_point_differential']
    y_test_clf = test_df['tennessee_won']
    
    # Train enhanced regression model
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    
    # Train enhanced classification model
    clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_model.fit(X_train, y_train_clf)
    
    # Make predictions
    reg_predictions = reg_model.predict(X_test)
    clf_predictions = clf_model.predict(X_test)
    clf_probabilities = clf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate enhanced models
    reg_mse = mean_squared_error(y_test_reg, reg_predictions)
    reg_r2 = r2_score(y_test_reg, reg_predictions)
    reg_mae = np.mean(np.abs(y_test_reg - reg_predictions))
    
    clf_accuracy = accuracy_score(y_test_clf, clf_predictions)
    
    print(f"Enhanced Model Performance:")
    print(f"   Regression R²: {reg_r2:.3f}")
    print(f"   Regression MAE: {reg_mae:.2f}")
    print(f"   Classification Accuracy: {clf_accuracy:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': reg_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top Enhanced Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Compare with previous model
    print(f"\n📊 Model Comparison:")
    print(f"   Previous R²: 0.760")
    print(f"   Enhanced R²: {reg_r2:.3f}")
    print(f"   Improvement: {reg_r2 - 0.760:+.3f}")
    
    print(f"   Previous Accuracy: 1.000")
    print(f"   Enhanced Accuracy: {clf_accuracy:.3f}")
    print(f"   Change: {clf_accuracy - 1.000:+.3f}")
    
    # Show 2024 predictions
    print(f"\n🎯 2024 Predictions with Enhanced Model:")
    print("-" * 50)
    
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        talent_tier = game['opponent_talent_tier']
        talent_score = game['opponent_talent']
        
        pred_diff = reg_predictions[i]
        pred_win = clf_predictions[i]
        pred_prob = clf_probabilities[i]
        
        actual_diff = game['tennessee_point_differential']
        actual_win = game['tennessee_won']
        
        home_away = "vs" if game['is_home_game'] else "@"
        win_indicator = "✅" if pred_win else "❌"
        actual_indicator = "✅" if actual_win else "❌"
        
        print(f"   Week {game['week']}: {home_away} {opponent} ({talent_tier}, {talent_score:.0f})")
        print(f"      Predicted: {win_indicator} {pred_diff:+.1f} points (Win prob: {pred_prob:.1%})")
        print(f"      Actual:    {actual_indicator} {actual_diff:+.1f} points")
        print(f"      Error:     {abs(pred_diff - actual_diff):.1f} points")
        print()
    
    # Model validation insights
    print(f"\n🔍 Key Insights:")
    print("-" * 50)
    
    # Analyze prediction accuracy by talent tier
    test_df['predicted_win'] = clf_predictions
    test_df['prediction_error'] = np.abs(reg_predictions - y_test_reg)
    test_df['win_prediction_correct'] = clf_predictions == y_test_clf
    
    tier_accuracy = test_df.groupby('opponent_talent_tier')['win_prediction_correct'].agg(['count', 'sum', 'mean'])
    print(f"Prediction accuracy by talent tier:")
    for tier, stats in tier_accuracy.iterrows():
        if stats['count'] > 0:
            print(f"   {tier}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Analyze prediction accuracy by talent difference
    test_df['talent_diff_tier'] = pd.cut(
        test_df['talent_difference'], 
        bins=[-200, -50, 50, 200], 
        labels=['Underdog', 'Even', 'Favorite']
    )
    
    diff_accuracy = test_df.groupby('talent_diff_tier')['win_prediction_correct'].agg(['count', 'sum', 'mean'])
    print(f"\nPrediction accuracy by talent difference:")
    for tier, stats in diff_accuracy.iterrows():
        if pd.notna(tier) and stats['count'] > 0:
            print(f"   {tier}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")

if __name__ == "__main__":
    main()
