#!/usr/bin/env python3
"""Enhanced ML model using comprehensive CFBD team and player stats."""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import requests
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Enhanced ML Model with Comprehensive CFBD Stats")
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
    
    # Fetch comprehensive stats
    print(f"\n🎯 Fetching comprehensive CFBD stats...")
    
    # 1. Team stats
    print(f"   📊 Fetching team stats...")
    team_stats = fetch_team_stats(api_key, headers, tennessee_df)
    
    # 2. Player stats
    print(f"   👥 Fetching player stats...")
    player_stats = fetch_player_stats(api_key, headers, tennessee_df)
    
    # 3. Advanced stats
    print(f"   📈 Fetching advanced stats...")
    advanced_stats = fetch_advanced_stats(api_key, headers, tennessee_df)
    
    # 4. Create comprehensive dataset
    print(f"\n🔗 Creating comprehensive enhanced dataset...")
    enhanced_df = create_comprehensive_dataset(tennessee_df, team_stats, player_stats, advanced_stats)
    
    # 5. Build enhanced ML model
    print(f"\n🤖 Building enhanced ML model...")
    build_enhanced_ml_model(enhanced_df)
    
    # Save comprehensive data
    filename = 'tennessee_games_comprehensive_stats.csv'
    enhanced_df.to_csv(filename, index=False)
    print(f"💾 Comprehensive data saved to: {filename}")

def fetch_team_stats(api_key, headers, tennessee_df):
    """Fetch comprehensive team stats."""
    
    team_stats = []
    
    try:
        for year in [2022, 2023, 2024]:
            print(f"      📊 Fetching {year} team stats...")
            
            # Team stats by season
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

def fetch_player_stats(api_key, headers, tennessee_df):
    """Fetch player stats."""
    
    player_stats = []
    
    try:
        for year in [2022, 2023, 2024]:
            print(f"      👥 Fetching {year} player stats...")
            
            # Player stats by season
            url = f'https://api.collegefootballdata.com/stats/player/season'
            params = {'year': year, 'seasonType': 'regular'}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                for player in data:
                    player_stats.append({
                        'year': year,
                        'player_id': player.get('playerId'),
                        'player_name': player.get('player'),
                        'team': player.get('team'),
                        'position': player.get('position'),
                        'games': player.get('games'),
                        'passing_yards': player.get('passingYards'),
                        'passing_tds': player.get('passingTds'),
                        'rushing_yards': player.get('rushingYards'),
                        'rushing_tds': player.get('rushingTds'),
                        'receiving_yards': player.get('receivingYards'),
                        'receiving_tds': player.get('receivingTds'),
                        'tackles': player.get('tackles'),
                        'sacks': player.get('sacks'),
                        'interceptions': player.get('interceptions')
                    })
                print(f"         ✅ Found {len(data)} players")
            else:
                print(f"         ❌ Error: {response.status_code}")
            
            time.sleep(0.5)
            
    except Exception as e:
        print(f"      ❌ Error fetching player stats: {e}")
    
    return pd.DataFrame(player_stats)

def fetch_advanced_stats(api_key, headers, tennessee_df):
    """Fetch advanced stats."""
    
    advanced_stats = []
    
    try:
        for year in [2022, 2023, 2024]:
            print(f"      📈 Fetching {year} advanced stats...")
            
            # Advanced stats by season
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

def create_comprehensive_dataset(tennessee_df, team_stats, player_stats, advanced_stats):
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
        
        # Get opponent player stats (top players)
        opponent_player_stats = player_stats[
            (player_stats['team'] == opponent) & 
            (player_stats['year'] == season)
        ]
        
        # Get Tennessee player stats (top players)
        tennessee_player_stats = player_stats[
            (player_stats['team'] == 'Tennessee') & 
            (player_stats['year'] == season)
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
        
        # Add player stats (top performers)
        if len(opponent_player_stats) > 0:
            # Convert to numeric
            opponent_player_stats['passing_yards'] = pd.to_numeric(opponent_player_stats['passing_yards'], errors='coerce').fillna(0)
            opponent_player_stats['rushing_yards'] = pd.to_numeric(opponent_player_stats['rushing_yards'], errors='coerce').fillna(0)
            opponent_player_stats['receiving_yards'] = pd.to_numeric(opponent_player_stats['receiving_yards'], errors='coerce').fillna(0)
            
            # Top QB
            qb_stats = opponent_player_stats[opponent_player_stats['position'] == 'QB'].nlargest(1, 'passing_yards')
            if len(qb_stats) > 0:
                enhanced_record['opponent_qb_yards'] = qb_stats.iloc[0]['passing_yards']
                enhanced_record['opponent_qb_tds'] = qb_stats.iloc[0]['passing_tds']
            else:
                enhanced_record['opponent_qb_yards'] = 0
                enhanced_record['opponent_qb_tds'] = 0
            
            # Top RB
            rb_stats = opponent_player_stats[opponent_player_stats['position'] == 'RB'].nlargest(1, 'rushing_yards')
            if len(rb_stats) > 0:
                enhanced_record['opponent_rb_yards'] = rb_stats.iloc[0]['rushing_yards']
                enhanced_record['opponent_rb_tds'] = rb_stats.iloc[0]['rushing_tds']
            else:
                enhanced_record['opponent_rb_yards'] = 0
                enhanced_record['opponent_rb_tds'] = 0
            
            # Top WR
            wr_stats = opponent_player_stats[opponent_player_stats['position'] == 'WR'].nlargest(1, 'receiving_yards')
            if len(wr_stats) > 0:
                enhanced_record['opponent_wr_yards'] = wr_stats.iloc[0]['receiving_yards']
                enhanced_record['opponent_wr_tds'] = wr_stats.iloc[0]['receiving_tds']
            else:
                enhanced_record['opponent_wr_yards'] = 0
                enhanced_record['opponent_wr_tds'] = 0
        else:
            enhanced_record.update({
                'opponent_qb_yards': 0, 'opponent_qb_tds': 0,
                'opponent_rb_yards': 0, 'opponent_rb_tds': 0,
                'opponent_wr_yards': 0, 'opponent_wr_tds': 0
            })
        
        if len(tennessee_player_stats) > 0:
            # Convert to numeric
            tennessee_player_stats['passing_yards'] = pd.to_numeric(tennessee_player_stats['passing_yards'], errors='coerce').fillna(0)
            tennessee_player_stats['rushing_yards'] = pd.to_numeric(tennessee_player_stats['rushing_yards'], errors='coerce').fillna(0)
            tennessee_player_stats['receiving_yards'] = pd.to_numeric(tennessee_player_stats['receiving_yards'], errors='coerce').fillna(0)
            
            # Top QB
            qb_stats = tennessee_player_stats[tennessee_player_stats['position'] == 'QB'].nlargest(1, 'passing_yards')
            if len(qb_stats) > 0:
                enhanced_record['tennessee_qb_yards'] = qb_stats.iloc[0]['passing_yards']
                enhanced_record['tennessee_qb_tds'] = qb_stats.iloc[0]['passing_tds']
            else:
                enhanced_record['tennessee_qb_yards'] = 0
                enhanced_record['tennessee_qb_tds'] = 0
            
            # Top RB
            rb_stats = tennessee_player_stats[tennessee_player_stats['position'] == 'RB'].nlargest(1, 'rushing_yards')
            if len(rb_stats) > 0:
                enhanced_record['tennessee_rb_yards'] = rb_stats.iloc[0]['rushing_yards']
                enhanced_record['tennessee_rb_tds'] = rb_stats.iloc[0]['rushing_tds']
            else:
                enhanced_record['tennessee_rb_yards'] = 0
                enhanced_record['tennessee_rb_tds'] = 0
            
            # Top WR
            wr_stats = tennessee_player_stats[tennessee_player_stats['position'] == 'WR'].nlargest(1, 'receiving_yards')
            if len(wr_stats) > 0:
                enhanced_record['tennessee_wr_yards'] = wr_stats.iloc[0]['receiving_yards']
                enhanced_record['tennessee_wr_tds'] = wr_stats.iloc[0]['receiving_tds']
            else:
                enhanced_record['tennessee_wr_yards'] = 0
                enhanced_record['tennessee_wr_tds'] = 0
        else:
            enhanced_record.update({
                'tennessee_qb_yards': 0, 'tennessee_qb_tds': 0,
                'tennessee_rb_yards': 0, 'tennessee_rb_tds': 0,
                'tennessee_wr_yards': 0, 'tennessee_wr_tds': 0
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

def build_enhanced_ml_model(df):
    """Build enhanced ML model with comprehensive stats."""
    
    print(f"\n🤖 Building Enhanced ML Model:")
    print("-" * 50)
    
    # Prepare features
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index',
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
        'opponent_qb_yards', 'opponent_qb_tds', 'opponent_rb_yards', 'opponent_rb_tds',
        'opponent_wr_yards', 'opponent_wr_tds',
        'tennessee_qb_yards', 'tennessee_qb_tds', 'tennessee_rb_yards', 'tennessee_rb_tds',
        'tennessee_wr_yards', 'tennessee_wr_tds',
        'ppg_difference', 'papg_difference', 'ypg_difference', 'yapg_difference',
        'turnover_difference', 'efficiency_difference'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in df.columns]
    
    print(f"📊 Using {len(available_features)} comprehensive features for ML")
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    reg_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    reg_model.fit(X_train_scaled, y_train_reg)
    
    clf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf_model.fit(X_train_scaled, y_train_clf)
    
    # Make predictions
    reg_predictions = reg_model.predict(X_test_scaled)
    clf_predictions = clf_model.predict(X_test_scaled)
    clf_probabilities = clf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate models
    reg_mse = mean_squared_error(y_test_reg, reg_predictions)
    reg_r2 = r2_score(y_test_reg, reg_predictions)
    reg_mae = np.mean(np.abs(y_test_reg - reg_predictions))
    
    clf_accuracy = accuracy_score(y_test_clf, clf_predictions)
    
    print(f"Comprehensive Stats Model Performance:")
    print(f"   Regression R²: {reg_r2:.3f}")
    print(f"   Regression MAE: {reg_mae:.2f}")
    print(f"   Classification Accuracy: {clf_accuracy:.3f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': reg_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top Comprehensive Features:")
    for _, row in importance_df.head(15).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Show 2024 predictions
    print(f"\n🎯 2024 Predictions with Comprehensive Stats Model:")
    print("-" * 50)
    
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        
        pred_diff = reg_predictions[i]
        pred_win = clf_predictions[i]
        pred_prob = clf_probabilities[i]
        
        actual_diff = game['tennessee_point_differential']
        actual_win = game['tennessee_won']
        
        home_away = "vs" if game['is_home_game'] else "@"
        win_indicator = "✅" if pred_win else "❌"
        actual_indicator = "✅" if actual_win else "❌"
        
        print(f"   Week {game['week']}: {home_away} {opponent}")
        print(f"      Predicted: {win_indicator} {pred_diff:+.1f} points (Win prob: {pred_prob:.1%})")
        print(f"      Actual:    {actual_indicator} {actual_diff:+.1f} points")
        print(f"      Error:     {abs(pred_diff - actual_diff):.1f} points")
        print()

if __name__ == "__main__":
    main()
