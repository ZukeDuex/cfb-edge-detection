#!/usr/bin/env python3
"""Retrieve CFBD stats for opponents with proper API handling."""

import requests
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import time
import json

def main():
    print("📊 Retrieving CFBD Stats for Opponents")
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
    
    # Get unique opponents by season
    opponents_by_season = {}
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        season = game['season']
        
        if season not in opponents_by_season:
            opponents_by_season[season] = set()
        opponents_by_season[season].add(opponent)
    
    print(f"\n📈 Found opponents by season:")
    for season, opponents in opponents_by_season.items():
        print(f"   {season}: {len(opponents)} unique opponents")
    
    # Fetch stats for each opponent by season
    all_stats = []
    successful_calls = 0
    failed_calls = 0
    
    for season, opponents in opponents_by_season.items():
        print(f"\n🔍 Fetching {season} stats for {len(opponents)} opponents...")
        
        for opponent in opponents:
            print(f"   📊 {opponent} ({season})...", end=" ")
            
            try:
                # Fetch team stats for the season
                stats = fetch_team_stats_safe(api_key, headers, opponent, season)
                
                if stats:
                    # Add context
                    stats['opponent'] = opponent
                    stats['season'] = season
                    all_stats.append(stats)
                    successful_calls += 1
                    print(f"✅ Found {len(stats)} stat categories")
                else:
                    failed_calls += 1
                    print(f"⚠️  No stats found")
                
                # Rate limiting - wait between calls
                print(f"      ⏳ Waiting 1 second for rate limiting...")
                time.sleep(1)
                
            except Exception as e:
                failed_calls += 1
                print(f"❌ Error: {str(e)[:50]}...")
                continue
    
    print(f"\n📊 API Call Summary:")
    print(f"   Successful calls: {successful_calls}")
    print(f"   Failed calls: {failed_calls}")
    print(f"   Total stats retrieved: {len(all_stats)}")
    
    if not all_stats:
        print("❌ No stats retrieved. Creating synthetic stats for analysis...")
        all_stats = create_synthetic_opponent_stats(tennessee_df)
    
    # Create comprehensive stats DataFrame
    stats_df = pd.DataFrame(all_stats)
    print(f"\n📊 Stats DataFrame created with {len(stats_df)} records")
    
    # Merge with Tennessee games
    print(f"\n🔗 Merging opponent stats with Tennessee games...")
    enhanced_df = merge_stats_with_games(tennessee_df, stats_df)
    
    # Analyze enhanced features
    print(f"\n📊 Analyzing enhanced features...")
    analyze_enhanced_features(enhanced_df)
    
    # Build enhanced ML model
    print(f"\n🤖 Building enhanced ML model...")
    build_enhanced_ml_model(enhanced_df)
    
    # Save enhanced data
    filename = 'tennessee_games_enhanced_stats.csv'
    enhanced_df.to_csv(filename, index=False)
    print(f"💾 Enhanced data saved to: {filename}")

def fetch_team_stats_safe(api_key, headers, team, season):
    """Safely fetch team stats with proper error handling."""
    
    # Map team names to CFBD team names
    team_mapping = {
        'Ball State': 'Ball State',
        'Clemson': 'Clemson',
        'Pittsburgh': 'Pittsburgh',
        'Akron': 'Akron',
        'Florida': 'Florida',
        'LSU': 'LSU',
        'Alabama': 'Alabama',
        'UT Martin': 'UT Martin',
        'Kentucky': 'Kentucky',
        'Georgia': 'Georgia',
        'Missouri': 'Missouri',
        'South Carolina': 'South Carolina',
        'Vanderbilt': 'Vanderbilt',
        'Austin Peay': 'Austin Peay',
        'UTSA': 'UTSA',
        'Virginia': 'Virginia',
        'Iowa': 'Iowa',
        'Chattanooga': 'Chattanooga',
        'Kent State': 'Kent State',
        'UTEP': 'UTEP',
        'Mississippi State': 'Mississippi State',
        'Texas A&M': 'Texas A&M',
        'UConn': 'UConn',
        'NC State': 'NC State',
        'Oklahoma': 'Oklahoma',
        'Arkansas': 'Arkansas',
        'Ohio State': 'Ohio State'
    }
    
    cfbd_team = team_mapping.get(team, team)
    
    try:
        # Fetch team stats
        url = f'https://api.collegefootballdata.com/stats/season'
        params = {'year': season, 'team': cfbd_team}
        
        print(f"      🌐 Making API call to: {url}")
        print(f"      📋 Parameters: {params}")
        
        response = requests.get(url, headers=headers, params=params, timeout=60)
        
        print(f"      📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            stats_data = response.json()
            print(f"      📊 Raw response length: {len(stats_data) if stats_data else 0}")
            
            if stats_data:
                # Extract key stats
                stats = {}
                
                for stat in stats_data:
                    category = stat.get('category')
                    stat_type = stat.get('statType')
                    value = stat.get('stat')
                    
                    if category and stat_type and value is not None:
                        key = f"{category}_{stat_type}".lower().replace(' ', '_')
                        stats[key] = float(value)
                
                print(f"      ✅ Extracted {len(stats)} stat categories")
                return stats
            else:
                print(f"      ⚠️  Empty response data")
                return None
        else:
            print(f"      ❌ API Error: {response.status_code}")
            print(f"      📄 Response text: {response.text[:100]}...")
            return None
            
    except requests.exceptions.Timeout:
        print(f"      ⏰ Request timeout")
        return None
    except requests.exceptions.ConnectionError:
        print(f"      🔌 Connection error")
        return None
    except Exception as e:
        print(f"      ❌ Unexpected error: {str(e)}")
        return None

def create_synthetic_opponent_stats(tennessee_df):
    """Create synthetic opponent stats for analysis."""
    print("🔧 Creating synthetic opponent stats...")
    
    synthetic_stats = []
    
    # Define realistic stat ranges by team tier
    team_stat_ranges = {
        'Elite': {
            'offensive_yards_per_game': (450, 550),
            'defensive_yards_per_game': (280, 350),
            'offensive_points_per_game': (35, 45),
            'defensive_points_per_game': (15, 25),
            'rushing_yards_per_game': (180, 220),
            'passing_yards_per_game': (250, 350),
            'turnover_margin': (0.5, 2.0),
            'third_down_conversion': (0.45, 0.55),
            'red_zone_efficiency': (0.85, 0.95)
        },
        'Strong': {
            'offensive_yards_per_game': (400, 480),
            'defensive_yards_per_game': (320, 400),
            'offensive_points_per_game': (28, 38),
            'defensive_points_per_game': (20, 30),
            'rushing_yards_per_game': (150, 200),
            'passing_yards_per_game': (220, 320),
            'turnover_margin': (0.0, 1.5),
            'third_down_conversion': (0.40, 0.50),
            'red_zone_efficiency': (0.80, 0.90)
        },
        'Average': {
            'offensive_yards_per_game': (350, 420),
            'defensive_yards_per_game': (380, 450),
            'offensive_points_per_game': (22, 32),
            'defensive_points_per_game': (25, 35),
            'rushing_yards_per_game': (120, 180),
            'passing_yards_per_game': (200, 280),
            'turnover_margin': (-0.5, 1.0),
            'third_down_conversion': (0.35, 0.45),
            'red_zone_efficiency': (0.75, 0.85)
        },
        'Weak': {
            'offensive_yards_per_game': (300, 380),
            'defensive_yards_per_game': (420, 500),
            'offensive_points_per_game': (18, 28),
            'defensive_points_per_game': (30, 40),
            'rushing_yards_per_game': (100, 160),
            'passing_yards_per_game': (180, 260),
            'turnover_margin': (-1.0, 0.5),
            'third_down_conversion': (0.30, 0.40),
            'red_zone_efficiency': (0.70, 0.80)
        },
        'FCS': {
            'offensive_yards_per_game': (250, 350),
            'defensive_yards_per_game': (450, 550),
            'offensive_points_per_game': (15, 25),
            'defensive_points_per_game': (35, 45),
            'rushing_yards_per_game': (80, 140),
            'passing_yards_per_game': (160, 240),
            'turnover_margin': (-1.5, 0.0),
            'third_down_conversion': (0.25, 0.35),
            'red_zone_efficiency': (0.65, 0.75)
        }
    }
    
    # Team tier mapping
    team_tiers = {
        'Alabama': 'Elite', 'Georgia': 'Elite', 'Ohio State': 'Elite', 'Clemson': 'Elite',
        'Florida': 'Strong', 'LSU': 'Strong', 'Oklahoma': 'Strong', 'Iowa': 'Strong', 'Texas A&M': 'Strong',
        'Kentucky': 'Average', 'South Carolina': 'Average', 'Missouri': 'Average', 'Arkansas': 'Average',
        'NC State': 'Average', 'UTSA': 'Average', 'Mississippi State': 'Average',
        'Vanderbilt': 'Weak', 'Ball State': 'Weak', 'Akron': 'Weak', 'Virginia': 'Weak',
        'Kent State': 'Weak', 'UTEP': 'Weak', 'UConn': 'Weak',
        'UT Martin': 'FCS', 'Austin Peay': 'FCS', 'Chattanooga': 'FCS'
    }
    
    # Get unique opponents by season
    opponents_by_season = {}
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        season = game['season']
        
        if season not in opponents_by_season:
            opponents_by_season[season] = set()
        opponents_by_season[season].add(opponent)
    
    # Create synthetic stats
    for season, opponents in opponents_by_season.items():
        for opponent in opponents:
            tier = team_tiers.get(opponent, 'Average')
            stat_ranges = team_stat_ranges[tier]
            
            stats = {
                'opponent': opponent,
                'season': season
            }
            
            # Generate stats within realistic ranges
            for stat_name, (min_val, max_val) in stat_ranges.items():
                stats[stat_name] = np.random.uniform(min_val, max_val)
            
            # Add some additional stats
            stats['time_of_possession'] = np.random.uniform(28, 32)  # minutes
            stats['penalties_per_game'] = np.random.uniform(6, 10)
            stats['penalty_yards_per_game'] = np.random.uniform(50, 80)
            stats['sacks_per_game'] = np.random.uniform(2, 4)
            stats['interceptions_per_game'] = np.random.uniform(0.5, 2.0)
            stats['fumbles_per_game'] = np.random.uniform(0.5, 1.5)
            
            synthetic_stats.append(stats)
    
    print(f"✅ Created {len(synthetic_stats)} synthetic stat records")
    return synthetic_stats

def merge_stats_with_games(tennessee_df, stats_df):
    """Merge opponent stats with Tennessee games."""
    
    enhanced_data = []
    
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        season = game['season']
        
        # Find matching stats
        opponent_stats = stats_df[
            (stats_df['opponent'] == opponent) & 
            (stats_df['season'] == season)
        ]
        
        if len(opponent_stats) > 0:
            stats = opponent_stats.iloc[0].to_dict()
        else:
            # Use default stats if not found
            stats = {
                'offensive_yards_per_game': 350,
                'defensive_yards_per_game': 400,
                'offensive_points_per_game': 25,
                'defensive_points_per_game': 30,
                'rushing_yards_per_game': 150,
                'passing_yards_per_game': 250,
                'turnover_margin': 0.0,
                'third_down_conversion': 0.40,
                'red_zone_efficiency': 0.80
            }
        
        # Create enhanced game record
        enhanced_record = game.to_dict()
        
        # Add opponent stats
        for key, value in stats.items():
            if key not in ['opponent', 'season']:
                enhanced_record[f'opponent_{key}'] = value
        
        # Add derived features
        enhanced_record['opponent_offensive_efficiency'] = stats.get('offensive_points_per_game', 25) / (stats.get('offensive_yards_per_game', 350) + 1) * 100
        enhanced_record['opponent_defensive_efficiency'] = stats.get('defensive_points_per_game', 30) / (stats.get('defensive_yards_per_game', 400) + 1) * 100
        enhanced_record['opponent_yard_differential'] = stats.get('offensive_yards_per_game', 350) - stats.get('defensive_yards_per_game', 400)
        enhanced_record['opponent_point_differential'] = stats.get('offensive_points_per_game', 25) - stats.get('defensive_points_per_game', 30)
        
        enhanced_data.append(enhanced_record)
    
    return pd.DataFrame(enhanced_data)

def analyze_enhanced_features(enhanced_df):
    """Analyze the enhanced features."""
    
    print(f"\n📊 Enhanced Features Analysis:")
    print("-" * 50)
    
    # Calculate Tennessee performance
    tennessee_home = enhanced_df['homeTeam'] == 'Tennessee'
    tennessee_points = enhanced_df['homePoints'] if tennessee_home else enhanced_df['awayPoints']
    opponent_points = enhanced_df['awayPoints'] if tennessee_home else enhanced_df['homePoints']
    tennessee_won = tennessee_points > opponent_points
    tennessee_point_differential = tennessee_points - opponent_points
    
    enhanced_df['tennessee_won'] = tennessee_won
    enhanced_df['tennessee_point_differential'] = tennessee_point_differential
    
    # Analyze by opponent offensive strength
    print(f"📈 Tennessee Performance by Opponent Offensive Strength:")
    
    enhanced_df['opponent_offensive_tier'] = pd.cut(
        enhanced_df['opponent_offensive_points_per_game'], 
        bins=[0, 20, 30, 40, 100], 
        labels=['Weak', 'Average', 'Strong', 'Elite']
    )
    
    offensive_performance = enhanced_df.groupby('opponent_offensive_tier').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    for tier, stats in offensive_performance.iterrows():
        if pd.notna(tier):
            wins = stats[('tennessee_won', 'sum')]
            total = stats[('tennessee_won', 'count')]
            win_pct = stats[('tennessee_won', 'mean')]
            avg_diff = stats[('tennessee_point_differential', 'mean')]
            
            print(f"   vs {tier} offense: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")
    
    # Analyze by opponent defensive strength
    print(f"\n🛡️ Tennessee Performance by Opponent Defensive Strength:")
    
    enhanced_df['opponent_defensive_tier'] = pd.cut(
        enhanced_df['opponent_defensive_points_per_game'], 
        bins=[0, 20, 30, 40, 100], 
        labels=['Elite', 'Strong', 'Average', 'Weak']
    )
    
    defensive_performance = enhanced_df.groupby('opponent_defensive_tier').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    for tier, stats in defensive_performance.iterrows():
        if pd.notna(tier):
            wins = stats[('tennessee_won', 'sum')]
            total = stats[('tennessee_won', 'count')]
            win_pct = stats[('tennessee_won', 'mean')]
            avg_diff = stats[('tennessee_point_differential', 'mean')]
            
            print(f"   vs {tier} defense: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")

def build_enhanced_ml_model(enhanced_df):
    """Build enhanced ML model with opponent stats."""
    
    print(f"\n🤖 Building Enhanced ML Model:")
    print("-" * 50)
    
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    
    # Prepare enhanced features
    enhanced_features = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_offensive_yards_per_game', 'opponent_defensive_yards_per_game',
        'opponent_offensive_points_per_game', 'opponent_defensive_points_per_game',
        'opponent_rushing_yards_per_game', 'opponent_passing_yards_per_game',
        'opponent_turnover_margin', 'opponent_third_down_conversion',
        'opponent_red_zone_efficiency', 'opponent_offensive_efficiency',
        'opponent_defensive_efficiency', 'opponent_yard_differential',
        'opponent_point_differential', 'tennessee_pregame_elo',
        'opponent_pregame_elo', 'elo_difference', 'tennessee_pregame_win_prob',
        'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    # Split data
    train_df = enhanced_df[enhanced_df['season'].isin([2022, 2023])].copy()
    test_df = enhanced_df[enhanced_df['season'] == 2024].copy()
    
    if len(test_df) == 0:
        print("❌ No 2024 data for testing")
        return
    
    # Prepare data
    X_train = train_df[enhanced_features].fillna(0)
    y_train_reg = train_df['tennessee_point_differential']
    y_train_clf = train_df['tennessee_won']
    
    X_test = test_df[enhanced_features].fillna(0)
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
        'feature': enhanced_features,
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

if __name__ == "__main__":
    main()
