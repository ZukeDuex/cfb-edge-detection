#!/usr/bin/env python3
"""Detailed Betting Analysis to identify profitable opportunities."""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import requests
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression, BayesianRidge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

def main():
    print("💰 Detailed Betting Analysis")
    print("=" * 50)
    
    # Load data
    try:
        tennessee_df = pd.read_csv('tennessee_games_2022_2024.csv')
        odds_df = pd.read_csv('tennessee_odds_2022_2024.csv')
        print(f"✅ Loaded {len(tennessee_df)} Tennessee games")
        print(f"✅ Loaded {len(odds_df)} betting lines")
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
    
    # Fetch comprehensive stats
    print(f"\n🎯 Fetching comprehensive CFBD stats...")
    
    # 1. Team stats
    print(f"   📊 Fetching team stats...")
    team_stats = fetch_team_stats(api_key, headers, tennessee_df)
    
    # 2. Advanced stats
    print(f"   📈 Fetching advanced stats...")
    advanced_stats = fetch_advanced_stats(api_key, headers, tennessee_df)
    
    # 3. Create comprehensive dataset
    print(f"\n🔗 Creating comprehensive enhanced dataset...")
    enhanced_df = create_comprehensive_dataset(tennessee_df, team_stats, advanced_stats)
    
    # 4. Error-focused feature engineering
    print(f"\n⚙️  Error-focused feature engineering...")
    engineered_df = error_focused_feature_engineering(enhanced_df)
    
    # 5. Build ML model
    print(f"\n🤖 Building ML model...")
    ml_model = build_ml_model(engineered_df)
    
    # 6. Detailed betting analysis
    print(f"\n💰 Detailed Betting Analysis:")
    print("-" * 50)
    
    analyze_betting_lines(engineered_df, odds_df, ml_model)

def fetch_team_stats(api_key, headers, tennessee_df):
    """Fetch comprehensive team stats."""
    
    team_stats = []
    
    try:
        for year in [2022, 2023, 2024]:
            print(f"      📊 Fetching {year} team stats...")
            
            url = f'https://api.collegefootballdata.com/stats/season'
            params = {'year': year, 'seasonType': 'regular'}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for team in data:
                    team_stats.append({
                        'year': year,
                        'team': team.get('team'),
                        'games': team.get('games'),
                        'wins': team.get('wins'),
                        'losses': team.get('losses'),
                        'ties': team.get('ties'),
                        'win_percentage': team.get('winPercentage'),
                        'points_per_game': team.get('pointsPerGame'),
                        'points_allowed_per_game': team.get('pointsAllowedPerGame'),
                        'yards_per_game': team.get('yardsPerGame'),
                        'yards_allowed_per_game': team.get('yardsAllowedPerGame'),
                        'passing_yards_per_game': team.get('passingYardsPerGame'),
                        'rushing_yards_per_game': team.get('rushingYardsPerGame'),
                        'passing_yards_allowed_per_game': team.get('passingYardsAllowedPerGame'),
                        'rushing_yards_allowed_per_game': team.get('rushingYardsAllowedPerGame'),
                        'turnovers': team.get('turnovers'),
                        'turnovers_forced': team.get('turnoversForced'),
                        'penalties': team.get('penalties'),
                        'penalty_yards': team.get('penaltyYards')
                    })
                print(f"         ✅ Found {len(data)} teams")
            else:
                print(f"         ❌ Error: {response.status_code}")
            
            time.sleep(0.5)
            
    except Exception as e:
        print(f"      ❌ Error fetching team stats: {e}")
    
    return pd.DataFrame(team_stats)

def fetch_advanced_stats(api_key, headers, tennessee_df):
    """Fetch advanced stats."""
    
    advanced_stats = []
    
    try:
        for year in [2022, 2023, 2024]:
            print(f"      📈 Fetching {year} advanced stats...")
            
            url = f'https://api.collegefootballdata.com/stats/season/advanced'
            params = {'year': year, 'seasonType': 'regular'}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for team in data:
                    advanced_stats.append({
                        'year': year,
                        'team': team.get('team'),
                        'offensive_efficiency': team.get('offensiveEfficiency'),
                        'defensive_efficiency': team.get('defensiveEfficiency'),
                        'special_teams_efficiency': team.get('specialTeamsEfficiency'),
                        'overall_efficiency': team.get('overallEfficiency'),
                        'offensive_explosiveness': team.get('offensiveExplosiveness'),
                        'defensive_explosiveness': team.get('defensiveExplosiveness'),
                        'offensive_field_position': team.get('offensiveFieldPosition'),
                        'defensive_field_position': team.get('defensiveFieldPosition'),
                        'offensive_havoc': team.get('offensiveHavoc'),
                        'defensive_havoc': team.get('defensiveHavoc'),
                        'offensive_passing_efficiency': team.get('offensivePassingEfficiency'),
                        'defensive_passing_efficiency': team.get('defensivePassingEfficiency'),
                        'offensive_rushing_efficiency': team.get('offensiveRushingEfficiency'),
                        'defensive_rushing_efficiency': team.get('defensiveRushingEfficiency')
                    })
                print(f"         ✅ Found {len(data)} teams")
            else:
                print(f"         ❌ Error: {response.status_code}")
            
            time.sleep(0.5)
            
    except Exception as e:
        print(f"      ❌ Error fetching advanced stats: {e}")
    
    return pd.DataFrame(advanced_stats)

def create_comprehensive_dataset(tennessee_df, team_stats, advanced_stats):
    """Create comprehensive enhanced dataset."""
    
    enhanced_data = []
    
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        season = game['season']
        
        # Get opponent team stats
        opponent_team_stats = team_stats[
            (team_stats['team'] == opponent) & 
            (team_stats['year'] == season)
        ]
        
        # Get Tennessee team stats
        tennessee_team_stats = team_stats[
            (team_stats['team'] == 'Tennessee') & 
            (team_stats['year'] == season)
        ]
        
        # Get opponent advanced stats
        opponent_advanced_stats = advanced_stats[
            (advanced_stats['team'] == opponent) & 
            (advanced_stats['year'] == season)
        ]
        
        # Get Tennessee advanced stats
        tennessee_advanced_stats = advanced_stats[
            (advanced_stats['team'] == 'Tennessee') & 
            (advanced_stats['year'] == season)
        ]
        
        # Create enhanced record
        enhanced_record = game.to_dict()
        
        # Add opponent team stats
        if len(opponent_team_stats) > 0:
            stats = opponent_team_stats.iloc[0]
            enhanced_record['opponent_games'] = stats.get('games', 0)
            enhanced_record['opponent_wins'] = stats.get('wins', 0)
            enhanced_record['opponent_losses'] = stats.get('losses', 0)
            enhanced_record['opponent_win_pct'] = stats.get('win_percentage', 0)
            enhanced_record['opponent_ppg'] = stats.get('points_per_game', 0)
            enhanced_record['opponent_papg'] = stats.get('points_allowed_per_game', 0)
            enhanced_record['opponent_ypg'] = stats.get('yards_per_game', 0)
            enhanced_record['opponent_yapg'] = stats.get('yards_allowed_per_game', 0)
            enhanced_record['opponent_pass_ypg'] = stats.get('passing_yards_per_game', 0)
            enhanced_record['opponent_rush_ypg'] = stats.get('rushing_yards_per_game', 0)
            enhanced_record['opponent_pass_yapg'] = stats.get('passing_yards_allowed_per_game', 0)
            enhanced_record['opponent_rush_yapg'] = stats.get('rushing_yards_allowed_per_game', 0)
            enhanced_record['opponent_turnovers'] = stats.get('turnovers', 0)
            enhanced_record['opponent_turnovers_forced'] = stats.get('turnovers_forced', 0)
            enhanced_record['opponent_penalties'] = stats.get('penalties', 0)
            enhanced_record['opponent_penalty_yards'] = stats.get('penalty_yards', 0)
        else:
            # Default values
            enhanced_record.update({
                'opponent_games': 0, 'opponent_wins': 0, 'opponent_losses': 0,
                'opponent_win_pct': 0, 'opponent_ppg': 0, 'opponent_papg': 0,
                'opponent_ypg': 0, 'opponent_yapg': 0, 'opponent_pass_ypg': 0,
                'opponent_rush_ypg': 0, 'opponent_pass_yapg': 0, 'opponent_rush_yapg': 0,
                'opponent_turnovers': 0, 'opponent_turnovers_forced': 0,
                'opponent_penalties': 0, 'opponent_penalty_yards': 0
            })
        
        # Add Tennessee team stats
        if len(tennessee_team_stats) > 0:
            stats = tennessee_team_stats.iloc[0]
            enhanced_record['tennessee_games'] = stats.get('games', 0)
            enhanced_record['tennessee_wins'] = stats.get('wins', 0)
            enhanced_record['tennessee_losses'] = stats.get('losses', 0)
            enhanced_record['tennessee_win_pct'] = stats.get('win_percentage', 0)
            enhanced_record['tennessee_ppg'] = stats.get('points_per_game', 0)
            enhanced_record['tennessee_papg'] = stats.get('points_allowed_per_game', 0)
            enhanced_record['tennessee_ypg'] = stats.get('yards_per_game', 0)
            enhanced_record['tennessee_yapg'] = stats.get('yards_allowed_per_game', 0)
            enhanced_record['tennessee_pass_ypg'] = stats.get('passing_yards_per_game', 0)
            enhanced_record['tennessee_rush_ypg'] = stats.get('rushing_yards_per_game', 0)
            enhanced_record['tennessee_pass_yapg'] = stats.get('passing_yards_allowed_per_game', 0)
            enhanced_record['tennessee_rush_yapg'] = stats.get('rushing_yards_allowed_per_game', 0)
            enhanced_record['tennessee_turnovers'] = stats.get('turnovers', 0)
            enhanced_record['tennessee_turnovers_forced'] = stats.get('turnovers_forced', 0)
            enhanced_record['tennessee_penalties'] = stats.get('penalties', 0)
            enhanced_record['tennessee_penalty_yards'] = stats.get('penalty_yards', 0)
        else:
            # Default values
            enhanced_record.update({
                'tennessee_games': 0, 'tennessee_wins': 0, 'tennessee_losses': 0,
                'tennessee_win_pct': 0, 'tennessee_ppg': 0, 'tennessee_papg': 0,
                'tennessee_ypg': 0, 'tennessee_yapg': 0, 'tennessee_pass_ypg': 0,
                'tennessee_rush_ypg': 0, 'tennessee_pass_yapg': 0, 'tennessee_rush_yapg': 0,
                'tennessee_turnovers': 0, 'tennessee_turnovers_forced': 0,
                'tennessee_penalties': 0, 'tennessee_penalty_yards': 0
            })
        
        # Add opponent advanced stats
        if len(opponent_advanced_stats) > 0:
            stats = opponent_advanced_stats.iloc[0]
            enhanced_record['opponent_off_eff'] = stats.get('offensive_efficiency', 0)
            enhanced_record['opponent_def_eff'] = stats.get('defensive_efficiency', 0)
            enhanced_record['opponent_special_eff'] = stats.get('special_teams_efficiency', 0)
            enhanced_record['opponent_overall_eff'] = stats.get('overall_efficiency', 0)
            enhanced_record['opponent_off_explosiveness'] = stats.get('offensive_explosiveness', 0)
            enhanced_record['opponent_def_explosiveness'] = stats.get('defensive_explosiveness', 0)
            enhanced_record['opponent_off_field_pos'] = stats.get('offensive_field_position', 0)
            enhanced_record['opponent_def_field_pos'] = stats.get('defensive_field_position', 0)
            enhanced_record['opponent_off_havoc'] = stats.get('offensive_havoc', 0)
            enhanced_record['opponent_def_havoc'] = stats.get('defensive_havoc', 0)
            enhanced_record['opponent_pass_eff'] = stats.get('offensive_passing_efficiency', 0)
            enhanced_record['opponent_pass_def_eff'] = stats.get('defensive_passing_efficiency', 0)
            enhanced_record['opponent_rush_eff'] = stats.get('offensive_rushing_efficiency', 0)
            enhanced_record['opponent_rush_def_eff'] = stats.get('defensive_rushing_efficiency', 0)
        else:
            # Default values
            enhanced_record.update({
                'opponent_off_eff': 0, 'opponent_def_eff': 0, 'opponent_special_eff': 0,
                'opponent_overall_eff': 0, 'opponent_off_explosiveness': 0, 'opponent_def_explosiveness': 0,
                'opponent_off_field_pos': 0, 'opponent_def_field_pos': 0, 'opponent_off_havoc': 0,
                'opponent_def_havoc': 0, 'opponent_pass_eff': 0, 'opponent_pass_def_eff': 0,
                'opponent_rush_eff': 0, 'opponent_rush_def_eff': 0
            })
        
        # Add Tennessee advanced stats
        if len(tennessee_advanced_stats) > 0:
            stats = tennessee_advanced_stats.iloc[0]
            enhanced_record['tennessee_off_eff'] = stats.get('offensive_efficiency', 0)
            enhanced_record['tennessee_def_eff'] = stats.get('defensive_efficiency', 0)
            enhanced_record['tennessee_special_eff'] = stats.get('special_teams_efficiency', 0)
            enhanced_record['tennessee_overall_eff'] = stats.get('overall_efficiency', 0)
            enhanced_record['tennessee_off_explosiveness'] = stats.get('offensive_explosiveness', 0)
            enhanced_record['tennessee_def_explosiveness'] = stats.get('defensive_explosiveness', 0)
            enhanced_record['tennessee_off_field_pos'] = stats.get('offensive_field_position', 0)
            enhanced_record['tennessee_def_field_pos'] = stats.get('defensive_field_position', 0)
            enhanced_record['tennessee_off_havoc'] = stats.get('offensive_havoc', 0)
            enhanced_record['tennessee_def_havoc'] = stats.get('defensive_havoc', 0)
            enhanced_record['tennessee_pass_eff'] = stats.get('offensive_passing_efficiency', 0)
            enhanced_record['tennessee_pass_def_eff'] = stats.get('defensive_passing_efficiency', 0)
            enhanced_record['tennessee_rush_eff'] = stats.get('offensive_rushing_efficiency', 0)
            enhanced_record['tennessee_rush_def_eff'] = stats.get('defensive_rushing_efficiency', 0)
        else:
            # Default values
            enhanced_record.update({
                'tennessee_off_eff': 0, 'tennessee_def_eff': 0, 'tennessee_special_eff': 0,
                'tennessee_overall_eff': 0, 'tennessee_off_explosiveness': 0, 'tennessee_def_explosiveness': 0,
                'tennessee_off_field_pos': 0, 'tennessee_def_field_pos': 0, 'tennessee_off_havoc': 0,
                'tennessee_def_havoc': 0, 'tennessee_pass_eff': 0, 'tennessee_pass_def_eff': 0,
                'tennessee_rush_eff': 0, 'tennessee_rush_def_eff': 0
            })
        
        # Calculate Tennessee performance
        tennessee_home = game['homeTeam'] == 'Tennessee'
        tennessee_points = game['homePoints'] if tennessee_home else game['awayPoints']
        opponent_points = game['awayPoints'] if tennessee_home else game['homePoints']
        tennessee_won = tennessee_points > opponent_points
        tennessee_point_differential = tennessee_points - opponent_points
        
        enhanced_record['tennessee_won'] = tennessee_won
        enhanced_record['tennessee_point_differential'] = tennessee_point_differential
        enhanced_record['is_home_game'] = tennessee_home
        
        # Add derived features (ensure numeric values)
        enhanced_record['ppg_difference'] = (enhanced_record.get('tennessee_ppg', 0) or 0) - (enhanced_record.get('opponent_ppg', 0) or 0)
        enhanced_record['papg_difference'] = (enhanced_record.get('opponent_papg', 0) or 0) - (enhanced_record.get('tennessee_papg', 0) or 0)
        enhanced_record['ypg_difference'] = (enhanced_record.get('tennessee_ypg', 0) or 0) - (enhanced_record.get('opponent_ypg', 0) or 0)
        enhanced_record['yapg_difference'] = (enhanced_record.get('opponent_yapg', 0) or 0) - (enhanced_record.get('tennessee_yapg', 0) or 0)
        enhanced_record['turnover_difference'] = (enhanced_record.get('tennessee_turnovers_forced', 0) or 0) - (enhanced_record.get('opponent_turnovers', 0) or 0)
        enhanced_record['efficiency_difference'] = (enhanced_record.get('tennessee_overall_eff', 0) or 0) - (enhanced_record.get('opponent_overall_eff', 0) or 0)
        
        enhanced_data.append(enhanced_record)
    
    return pd.DataFrame(enhanced_data)

def error_focused_feature_engineering(df):
    """Error-focused feature engineering to minimize prediction error."""
    
    print(f"   🔧 Creating error-minimizing features...")
    
    engineered_df = df.copy()
    
    # 1. Error-focused interaction features
    engineered_df['elo_home_interaction'] = (engineered_df['homePregameElo'] - engineered_df['awayPregameElo']) * engineered_df['is_home_game']
    engineered_df['week_home_interaction'] = engineered_df['week'] * engineered_df['is_home_game']
    engineered_df['attendance_home_interaction'] = engineered_df['attendance'] * engineered_df['is_home_game']
    engineered_df['win_prob_home_interaction'] = engineered_df['homePostgameWinProbability'] * engineered_df['is_home_game']
    
    # 2. Polynomial features for non-linear relationships
    engineered_df['week_squared'] = engineered_df['week'] ** 2
    engineered_df['week_cubed'] = engineered_df['week'] ** 3
    engineered_df['elo_difference_squared'] = (engineered_df['homePregameElo'] - engineered_df['awayPregameElo']) ** 2
    engineered_df['attendance_squared'] = engineered_df['attendance'] ** 2
    engineered_df['attendance_log'] = np.log(engineered_df['attendance'] + 1)
    
    # 3. Ratio features for relative strength
    engineered_df['ppg_ratio'] = engineered_df['tennessee_ppg'] / (engineered_df['opponent_ppg'] + 1)
    engineered_df['papg_ratio'] = engineered_df['opponent_papg'] / (engineered_df['tennessee_papg'] + 1)
    engineered_df['ypg_ratio'] = engineered_df['tennessee_ypg'] / (engineered_df['opponent_ypg'] + 1)
    engineered_df['yapg_ratio'] = engineered_df['opponent_yapg'] / (engineered_df['tennessee_yapg'] + 1)
    engineered_df['elo_ratio'] = engineered_df['homePregameElo'] / (engineered_df['awayPregameElo'] + 1)
    
    # 4. Time-based features with error focus
    engineered_df['is_early_season'] = engineered_df['week'] <= 3
    engineered_df['is_mid_season'] = (engineered_df['week'] > 3) & (engineered_df['week'] < 9)
    engineered_df['is_late_season'] = engineered_df['week'] >= 9
    engineered_df['is_postseason'] = engineered_df['seasonType'] == 'postseason'
    engineered_df['week_from_end'] = 15 - engineered_df['week']  # Weeks remaining
    
    # 5. Momentum features (rolling averages)
    engineered_df = engineered_df.sort_values(['season', 'week'])
    engineered_df['tennessee_points_rolling'] = engineered_df.groupby('season')['tennessee_point_differential'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    engineered_df['opponent_points_rolling'] = engineered_df.groupby('season')['opponent_ppg'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    engineered_df['tennessee_win_rolling'] = engineered_df.groupby('season')['tennessee_won'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # 6. Conference strength features
    conference_strength = {
        'SEC': 0.9, 'Big Ten': 0.85, 'Big 12': 0.8, 'ACC': 0.75, 'Pac-12': 0.7,
        'American Athletic': 0.6, 'Mountain West': 0.55, 'MAC': 0.5, 'Sun Belt': 0.45,
        'Conference USA': 0.4, 'FBS Independents': 0.6, 'Big South-OVC': 0.3,
        'Southern': 0.2, 'UAC': 0.25
    }
    engineered_df['conference_strength'] = engineered_df['awayConference'].map(conference_strength).fillna(0.5)
    
    # 7. Weather and venue features
    engineered_df['is_cold_weather'] = engineered_df['week'] >= 10
    engineered_df['is_warm_weather'] = engineered_df['week'] <= 4
    engineered_df['is_modern_venue'] = engineered_df['venueId'] > 1000
    engineered_df['is_old_venue'] = engineered_df['venueId'] < 500
    
    # 8. Error-specific features
    engineered_df['elo_difference'] = engineered_df['homePregameElo'] - engineered_df['awayPregameElo']
    engineered_df['win_prob_difference'] = engineered_df['homePostgameWinProbability'] - engineered_df['awayPostgameWinProbability']
    engineered_df['excitement_factor'] = engineered_df['excitementIndex'] * engineered_df['attendance']
    
    # 9. Advanced statistical features
    engineered_df['tennessee_consistency'] = 1 / (engineered_df['tennessee_ppg'] - engineered_df['tennessee_papg'] + 1)
    engineered_df['opponent_consistency'] = 1 / (engineered_df['opponent_ppg'] - engineered_df['opponent_papg'] + 1)
    engineered_df['relative_strength'] = (engineered_df['tennessee_ppg'] - engineered_df['tennessee_papg']) - (engineered_df['opponent_ppg'] - engineered_df['opponent_papg'])
    
    print(f"      ✅ Created {len(engineered_df.columns) - len(df.columns)} error-minimizing features")
    
    return engineered_df

def build_ml_model(df):
    """Build ML model for predictions."""
    
    print(f"\n🤖 Building ML Model:")
    print("-" * 30)
    
    # Prepare features
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'homePregameElo', 'awayPregameElo', 'homePostgameWinProbability', 'awayPostgameWinProbability', 'excitement_index',
        'opponent_games', 'opponent_wins', 'opponent_losses', 'opponent_win_pct',
        'opponent_ppg', 'opponent_papg', 'opponent_ypg', 'opponent_yapg',
        'opponent_pass_ypg', 'opponent_rush_ypg', 'opponent_pass_yapg', 'opponent_rush_yapg',
        'opponent_turnovers', 'opponent_turnovers_forced', 'opponent_penalties', 'opponent_penalty_yards',
        'tennessee_games', 'tennessee_wins', 'tennessee_losses', 'tennessee_win_pct',
        'tennessee_ppg', 'tennessee_papg', 'tennessee_ypg', 'tennessee_yapg',
        'tennessee_pass_ypg', 'tennessee_rush_ypg', 'tennessee_pass_yapg', 'tennessee_rush_yapg',
        'tennessee_turnovers', 'tennessee_turnovers_forced', 'tennessee_penalties', 'tennessee_penalty_yards',
        'opponent_off_eff', 'opponent_def_eff', 'opponent_special_eff', 'opponent_overall_eff',
        'opponent_off_explosiveness', 'opponent_def_explosiveness', 'opponent_off_field_pos', 'opponent_def_field_pos',
        'opponent_off_havoc', 'opponent_def_havoc', 'opponent_pass_eff', 'opponent_pass_def_eff',
        'opponent_rush_eff', 'opponent_rush_def_eff',
        'tennessee_off_eff', 'tennessee_def_eff', 'tennessee_special_eff', 'tennessee_overall_eff',
        'tennessee_off_explosiveness', 'tennessee_def_explosiveness', 'tennessee_off_field_pos', 'tennessee_def_field_pos',
        'tennessee_off_havoc', 'tennessee_def_havoc', 'tennessee_pass_eff', 'tennessee_pass_def_eff',
        'tennessee_rush_eff', 'tennessee_rush_def_eff',
        'ppg_difference', 'papg_difference', 'ypg_difference', 'yapg_difference',
        'turnover_difference', 'efficiency_difference',
        'elo_home_interaction', 'week_home_interaction', 'attendance_home_interaction', 'win_prob_home_interaction',
        'week_squared', 'week_cubed', 'elo_difference_squared', 'attendance_squared', 'attendance_log',
        'ppg_ratio', 'papg_ratio', 'ypg_ratio', 'yapg_ratio', 'elo_ratio',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason', 'week_from_end',
        'tennessee_points_rolling', 'opponent_points_rolling', 'tennessee_win_rolling', 'conference_strength',
        'is_cold_weather', 'is_warm_weather', 'is_modern_venue', 'is_old_venue',
        'elo_difference', 'win_prob_difference', 'excitement_factor',
        'tennessee_consistency', 'opponent_consistency', 'relative_strength'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in df.columns]
    
    print(f"📊 Using {len(available_features)} features for ML")
    
    # Split data
    train_df = df[df['season'].isin([2022, 2023])].copy()
    test_df = df[df['season'] == 2024].copy()
    
    if len(test_df) == 0:
        print("❌ No 2024 data for testing")
        return None
    
    # Prepare data
    X_train = train_df[available_features].fillna(0)
    y_train_reg = train_df['tennessee_point_differential']
    y_train_clf = train_df['tennessee_won']
    
    X_test = test_df[available_features].fillna(0)
    y_test_reg = test_df['tennessee_point_differential']
    y_test_clf = test_df['tennessee_won']
    
    # Feature selection
    selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(available_features)))
    X_train_selected = selector.fit_transform(X_train, y_train_reg)
    X_test_selected = selector.transform(X_test)
    
    selected_features = [available_features[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} top features")
    
    # Scale features
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Build stacking ensemble
    stacking_reg = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=2, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=10, random_state=42)),
            ('ridge', Ridge(alpha=0.1)),
            ('bayesian', BayesianRidge())
        ],
        final_estimator=Ridge(alpha=0.1),
        cv=5
    )
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=2, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=10, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=2000))
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # Train models
    stacking_reg.fit(X_train_scaled, y_train_reg)
    stacking_clf.fit(X_train_scaled, y_train_clf)
    
    # Make predictions
    reg_predictions = stacking_reg.predict(X_test_scaled)
    clf_predictions = stacking_clf.predict(X_test_scaled)
    clf_probabilities = stacking_clf.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate models
    reg_mae = mean_absolute_error(y_test_reg, reg_predictions)
    reg_r2 = r2_score(y_test_reg, reg_predictions)
    clf_accuracy = accuracy_score(y_test_clf, clf_predictions)
    
    print(f"ML Model Performance:")
    print(f"   Regression MAE: {reg_mae:.2f}, R²: {reg_r2:.3f}")
    print(f"   Classification Accuracy: {clf_accuracy:.3f}")
    
    return {
        'reg_model': stacking_reg,
        'clf_model': stacking_clf,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'reg_predictions': reg_predictions,
        'clf_predictions': clf_predictions,
        'clf_probabilities': clf_probabilities,
        'test_df': test_df
    }

def analyze_betting_lines(df, odds_df, ml_model):
    """Analyze betting lines and identify profitable opportunities."""
    
    print(f"   📊 Processing betting lines data...")
    
    # Filter for spread bets only
    spread_odds = odds_df[odds_df['market'] == 'spreads'].copy()
    
    # Group by game and get the best line for each team
    game_lines = {}
    
    for _, row in spread_odds.iterrows():
        game_id = row['game_id']
        team = row['outcome']
        point = row['point']
        price = row['price']
        
        if game_id not in game_lines:
            game_lines[game_id] = {}
        
        # Store the best line (most favorable for bettor)
        if team not in game_lines[game_id] or abs(point) > abs(game_lines[game_id][team]['point']):
            game_lines[game_id][team] = {
                'point': point,
                'price': price,
                'book': row['book']
            }
    
    print(f"      ✅ Processed {len(game_lines)} games with betting lines")
    
    # Analyze betting opportunities
    print(f"\n🎯 Detailed Betting Analysis:")
    print("-" * 30)
    
    profitable_bets = []
    total_profit = 0
    total_bets = 0
    
    test_df = ml_model['test_df']
    reg_predictions = ml_model['reg_predictions']
    clf_predictions = ml_model['clf_predictions']
    clf_probabilities = ml_model['clf_probabilities']
    
    for i, (_, game) in enumerate(test_df.iterrows()):
        game_id = str(game['id'])
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        
        # Get ML predictions
        ml_pred_diff = reg_predictions[i]
        ml_pred_win = clf_predictions[i]
        ml_pred_prob = clf_probabilities[i]
        
        # Get actual result
        actual_diff = game['tennessee_point_differential']
        actual_win = game['tennessee_won']
        
        # Get betting lines
        if game_id in game_lines:
            lines = game_lines[game_id]
            
            # Find Tennessee line
            tennessee_line = None
            tennessee_price = None
            tennessee_book = None
            
            for team, line_info in lines.items():
                if 'Tennessee' in team:
                    tennessee_line = line_info['point']
                    tennessee_price = line_info['price']
                    tennessee_book = line_info['book']
                    break
            
            if tennessee_line is not None:
                # Calculate betting edge
                ml_edge = ml_pred_diff - tennessee_line
                
                # Determine bet recommendation
                bet_recommendation = "NO BET"
                bet_amount = 0
                expected_profit = 0
                
                if abs(ml_edge) > 2:  # Lower threshold for more opportunities
                    if ml_edge > 0:  # ML predicts Tennessee covers
                        bet_recommendation = "BET TENNESSEE"
                        bet_amount = min(100, abs(ml_edge) * 15)  # Bet more on larger edges
                        expected_profit = bet_amount * (ml_pred_prob - 0.5) * 2
                    else:  # ML predicts Tennessee doesn't cover
                        bet_recommendation = "BET OPPONENT"
                        bet_amount = min(100, abs(ml_edge) * 15)
                        expected_profit = bet_amount * ((1 - ml_pred_prob) - 0.5) * 2
                
                # Calculate actual profit/loss
                actual_profit = 0
                if bet_recommendation != "NO BET":
                    if bet_recommendation == "BET TENNESSEE":
                        if actual_win and actual_diff > tennessee_line:
                            actual_profit = bet_amount * 0.91  # -110 odds
                        else:
                            actual_profit = -bet_amount
                    else:  # BET OPPONENT
                        if not actual_win or actual_diff < tennessee_line:
                            actual_profit = bet_amount * 0.91  # -110 odds
                        else:
                            actual_profit = -bet_amount
                
                total_profit += actual_profit
                total_bets += 1
                
                profitable_bets.append({
                    'game_id': game_id,
                    'opponent': opponent,
                    'week': game['week'],
                    'tennessee_line': tennessee_line,
                    'ml_pred_diff': ml_pred_diff,
                    'ml_edge': ml_edge,
                    'ml_pred_prob': ml_pred_prob,
                    'bet_recommendation': bet_recommendation,
                    'bet_amount': bet_amount,
                    'expected_profit': expected_profit,
                    'actual_diff': actual_diff,
                    'actual_win': actual_win,
                    'actual_profit': actual_profit,
                    'book': tennessee_book
                })
                
                # Print betting opportunity
                home_away = "vs" if game['is_home_game'] else "@"
                print(f"   Week {game['week']}: {home_away} {opponent}")
                print(f"      Line: {tennessee_line:+.1f} ({tennessee_book})")
                print(f"      ML Pred: {ml_pred_diff:+.1f} (Win prob: {ml_pred_prob:.1%})")
                print(f"      Edge: {ml_edge:+.1f} points")
                print(f"      Recommendation: {bet_recommendation}")
                if bet_amount > 0:
                    print(f"      Bet Amount: ${bet_amount:.0f}")
                    print(f"      Expected Profit: ${expected_profit:+.2f}")
                    print(f"      Actual Profit: ${actual_profit:+.2f}")
                print()
    
    # Calculate betting strategy performance
    print(f"\n📊 Betting Strategy Performance:")
    print("-" * 30)
    
    if total_bets > 0:
        roi = (total_profit / (total_bets * 100)) * 100  # Assuming $100 average bet
        win_rate = len([b for b in profitable_bets if b['actual_profit'] > 0]) / total_bets
        
        print(f"   Total Bets: {total_bets}")
        print(f"   Total Profit: ${total_profit:+.2f}")
        print(f"   ROI: {roi:+.2f}%")
        print(f"   Win Rate: {win_rate:.1%}")
        
        # Analyze by bet type
        tennessee_bets = [b for b in profitable_bets if b['bet_recommendation'] == 'BET TENNESSEE']
        opponent_bets = [b for b in profitable_bets if b['bet_recommendation'] == 'BET OPPONENT']
        
        if tennessee_bets:
            tennessee_profit = sum(b['actual_profit'] for b in tennessee_bets)
            tennessee_win_rate = len([b for b in tennessee_bets if b['actual_profit'] > 0]) / len(tennessee_bets)
            print(f"   Tennessee Bets: {len(tennessee_bets)} | Profit: ${tennessee_profit:+.2f} | Win Rate: {tennessee_win_rate:.1%}")
        
        if opponent_bets:
            opponent_profit = sum(b['actual_profit'] for b in opponent_bets)
            opponent_win_rate = len([b for b in opponent_bets if b['actual_profit'] > 0]) / len(opponent_bets)
            print(f"   Opponent Bets: {len(opponent_bets)} | Profit: ${opponent_profit:+.2f} | Win Rate: {opponent_win_rate:.1%}")
        
        # Analyze by edge size
        large_edge_bets = [b for b in profitable_bets if abs(b['ml_edge']) > 5]
        medium_edge_bets = [b for b in profitable_bets if 2 < abs(b['ml_edge']) <= 5]
        
        if large_edge_bets:
            large_edge_profit = sum(b['actual_profit'] for b in large_edge_bets)
            large_edge_win_rate = len([b for b in large_edge_bets if b['actual_profit'] > 0]) / len(large_edge_bets)
            print(f"   Large Edge (>5): {len(large_edge_bets)} | Profit: ${large_edge_profit:+.2f} | Win Rate: {large_edge_win_rate:.1%}")
        
        if medium_edge_bets:
            medium_edge_profit = sum(b['actual_profit'] for b in medium_edge_bets)
            medium_edge_win_rate = len([b for b in medium_edge_bets if b['actual_profit'] > 0]) / len(medium_edge_bets)
            print(f"   Medium Edge (2-5): {len(medium_edge_bets)} | Profit: ${medium_edge_profit:+.2f} | Win Rate: {medium_edge_win_rate:.1%}")
    
    # Save profitable bets
    if profitable_bets:
        bets_df = pd.DataFrame(profitable_bets)
        bets_df.to_csv('profitable_bets_2024.csv', index=False)
        print(f"💾 Profitable bets saved to: profitable_bets_2024.csv")
    
    # Betting recommendations
    print(f"\n🎯 Betting Recommendations:")
    print("-" * 30)
    
    if total_profit > 0:
        print(f"✅ PROFITABLE STRATEGY IDENTIFIED!")
        print(f"   Focus on bets with edge > 2 points")
        print(f"   Bet more on larger edges (5+ points)")
        print(f"   Target home games with clear talent advantages")
    else:
        print(f"❌ Strategy needs improvement")
        print(f"   Consider higher edge thresholds")
        print(f"   Focus on most confident predictions")
        print(f"   Avoid betting on close games")

if __name__ == "__main__":
    main()
