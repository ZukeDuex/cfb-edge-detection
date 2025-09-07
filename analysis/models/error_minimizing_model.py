#!/usr/bin/env python3
"""Error-Minimizing ML Model with Advanced Techniques."""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import requests
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, RANSACRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, validation_curve
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🎯 Error-Minimizing ML Model")
    print("=" * 50)
    
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
    
    # 5. Build error-minimizing ML model
    print(f"\n🤖 Building error-minimizing ML model...")
    build_error_minimizing_model(engineered_df)
    
    # Save comprehensive data
    filename = 'tennessee_games_error_minimized.csv'
    engineered_df.to_csv(filename, index=False)
    print(f"💾 Error-minimized data saved to: {filename}")

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
                'tennessee_win_pct': 0, 'tennessee_papg': 0, 'tennessee_ppg': 0,
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

def build_error_minimizing_model(df):
    """Build error-minimizing ML model with advanced techniques."""
    
    print(f"\n🤖 Building Error-Minimizing ML Model:")
    print("-" * 50)
    
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
    
    print(f"📊 Using {len(available_features)} features for error minimization")
    
    # Split data
    train_df = df[df['season'].isin([2022, 2023])].copy()
    test_df = df[df['season'] == 2024].copy()
    
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
    
    # Advanced feature selection for error minimization
    print(f"\n🔍 Error-Focused Feature Selection:")
    print("-" * 30)
    
    # Select top features using multiple methods
    selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(available_features)))
    X_train_selected = selector.fit_transform(X_train, y_train_reg)
    X_test_selected = selector.transform(X_test)
    
    selected_features = [available_features[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} top features for error minimization:")
    for feature in selected_features:
        print(f"   {feature}")
    
    # Advanced scaling for error minimization
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Build error-minimizing ensemble models
    print(f"\n🎯 Building Error-Minimizing Ensemble Models:")
    print("-" * 30)
    
    # Regression models optimized for error minimization
    reg_models = {
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=10, min_samples_split=2, random_state=42),
        'Ridge': Ridge(alpha=0.1),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'Bayesian Ridge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
        'Huber Regressor': HuberRegressor(epsilon=1.35, max_iter=200),
        'RANSAC Regressor': RANSACRegressor(random_state=42),
        'SVR': SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(200, 100, 50), max_iter=2000, learning_rate='adaptive', random_state=42)
    }
    
    # Classification models optimized for accuracy
    clf_models = {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=10, min_samples_split=2, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, C=10.0),
        'SVC': SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=2000, learning_rate='adaptive', random_state=42)
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
            medae = median_absolute_error(y_test_reg, y_pred)
            r2 = r2_score(y_test_reg, y_pred)
            
            reg_results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'mae': mae,
                'medae': medae,
                'r2': r2
            }
            
            print(f"      MAE: {mae:.2f}, MedAE: {medae:.2f}, R²: {r2:.3f}")
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
    
    # Build stacking ensemble for error minimization
    print(f"\n🗳️  Building Stacking Ensemble for Error Minimization:")
    print("-" * 30)
    
    # Stacking regressor for error minimization
    stacking_reg = StackingRegressor(
        estimators=[
            ('rf', reg_results['Random Forest']['model']),
            ('gb', reg_results['Gradient Boosting']['model']),
            ('ridge', reg_results['Ridge']['model']),
            ('huber', reg_results['Huber Regressor']['model'])
        ],
        final_estimator=Ridge(alpha=0.1),
        cv=5
    )
    
    stacking_reg.fit(X_train_scaled, y_train_reg)
    stacking_reg_pred = stacking_reg.predict(X_test_scaled)
    
    stacking_reg_mse = mean_squared_error(y_test_reg, stacking_reg_pred)
    stacking_reg_mae = mean_absolute_error(y_test_reg, stacking_reg_pred)
    stacking_reg_medae = median_absolute_error(y_test_reg, stacking_reg_pred)
    stacking_reg_r2 = r2_score(y_test_reg, stacking_reg_pred)
    
    print(f"   Stacking Regression:")
    print(f"      MAE: {stacking_reg_mae:.2f}, MedAE: {stacking_reg_medae:.2f}, R²: {stacking_reg_r2:.3f}")
    
    # Stacking classifier for accuracy
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
    
    print(f"   Stacking Classification:")
    print(f"      Accuracy: {stacking_clf_accuracy:.3f}")
    
    # Hyperparameter tuning for error minimization
    print(f"\n⚙️  Hyperparameter Tuning for Error Minimization:")
    print("-" * 30)
    
    # Find best model for error minimization
    best_reg_model = min(reg_results.items(), key=lambda x: x[1]['mae'])
    best_clf_model = max(clf_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"   Best Regression Model: {best_reg_model[0]} (MAE: {best_reg_model[1]['mae']:.2f})")
    print(f"   Best Classification Model: {best_clf_model[0]} (Accuracy: {best_clf_model[1]['accuracy']:.3f})")
    
    # Cross-validation for error minimization
    print(f"\n🔄 Cross-Validation for Error Minimization:")
    print("-" * 30)
    
    cv_scores_reg = cross_val_score(best_reg_model[1]['model'], X_train_scaled, y_train_reg, cv=5, scoring='neg_mean_absolute_error')
    cv_scores_clf = cross_val_score(best_clf_model[1]['model'], X_train_scaled, y_train_clf, cv=5, scoring='accuracy')
    
    print(f"   Cross-validation MAE: {-cv_scores_reg.mean():.2f} (+/- {cv_scores_reg.std() * 2:.2f})")
    print(f"   Cross-validation Accuracy: {cv_scores_clf.mean():.3f} (+/- {cv_scores_clf.std() * 2:.3f})")
    
    # Model comparison focused on error minimization
    print(f"\n📊 Error-Minimizing Model Comparison:")
    print("-" * 30)
    
    print(f"   Regression Models (sorted by MAE):")
    sorted_reg = sorted(reg_results.items(), key=lambda x: x[1]['mae'])
    for name, results in sorted_reg:
        print(f"      {name}: MAE = {results['mae']:.2f}, MedAE = {results['medae']:.2f}, R² = {results['r2']:.3f}")
    
    print(f"   Classification Models (sorted by accuracy):")
    sorted_clf = sorted(clf_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, results in sorted_clf:
        print(f"      {name}: Accuracy = {results['accuracy']:.3f}")
    
    print(f"   Ensemble Models:")
    print(f"      Stacking Regression: MAE = {stacking_reg_mae:.2f}, MedAE = {stacking_reg_medae:.2f}, R² = {stacking_reg_r2:.3f}")
    print(f"      Stacking Classification: Accuracy = {stacking_clf_accuracy:.3f}")
    
    # Show 2024 predictions with error-minimizing models
    print(f"\n🎯 2024 Predictions with Error-Minimizing Models:")
    print("-" * 50)
    
    best_reg_pred = best_reg_model[1]['predictions']
    best_clf_pred = best_clf_model[1]['predictions']
    best_clf_prob = best_clf_model[1]['probabilities']
    
    total_error = 0
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        
        pred_diff = best_reg_pred[i]
        pred_win = best_clf_pred[i]
        pred_prob = best_clf_prob[i]
        
        actual_diff = game['tennessee_point_differential']
        actual_win = game['tennessee_won']
        
        error = abs(pred_diff - actual_diff)
        total_error += error
        
        home_away = "vs" if game['is_home_game'] else "@"
        win_indicator = "✅" if pred_win else "❌"
        actual_indicator = "✅" if actual_win else "❌"
        
        print(f"   Week {game['week']}: {home_away} {opponent}")
        print(f"      Predicted: {win_indicator} {pred_diff:+.1f} points (Win prob: {pred_prob:.1%})")
        print(f"      Actual:    {actual_indicator} {actual_diff:+.1f} points")
        print(f"      Error:     {error:.1f} points")
        print()
    
    avg_error = total_error / len(test_df)
    print(f"📊 Average Prediction Error: {avg_error:.2f} points")
    print(f"📊 Total Prediction Error: {total_error:.2f} points")

if __name__ == "__main__":
    main()
