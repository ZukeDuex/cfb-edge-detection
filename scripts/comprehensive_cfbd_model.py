#!/usr/bin/env python3
"""Comprehensive ML model leveraging all CFBD stats: team info, matchups, rosters, and talent."""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import requests
import time

def main():
    print("📊 Comprehensive ML Model with All CFBD Stats")
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
    
    # Fetch comprehensive data
    print(f"\n🎯 Fetching comprehensive CFBD data...")
    
    # 1. Team talent data
    print(f"   📊 Fetching team talent data...")
    talent_data = fetch_team_talent_data(api_key, headers)
    
    # 2. Team info data
    print(f"   🏟️  Fetching team info data...")
    team_info_data = fetch_team_info_data(api_key, headers)
    
    # 3. Team matchup history
    print(f"   📈 Fetching team matchup history...")
    matchup_data = fetch_team_matchup_data(api_key, headers, tennessee_df)
    
    # 4. Create comprehensive enhanced dataset
    print(f"\n🔗 Creating comprehensive enhanced dataset...")
    enhanced_df = create_comprehensive_dataset(tennessee_df, talent_data, team_info_data, matchup_data)
    
    # Analyze comprehensive features
    print(f"\n📊 Analyzing comprehensive features...")
    analyze_comprehensive_features(enhanced_df)
    
    # Build comprehensive ML model
    print(f"\n🤖 Building comprehensive ML model...")
    build_comprehensive_ml_model(enhanced_df)
    
    # Save comprehensive data
    filename = 'tennessee_games_comprehensive.csv'
    enhanced_df.to_csv(filename, index=False)
    print(f"💾 Comprehensive data saved to: {filename}")

def fetch_team_talent_data(api_key, headers):
    """Fetch team talent data."""
    
    talent_data = []
    
    try:
        for year in [2022, 2023, 2024]:
            print(f"      📊 Fetching {year} talent data...")
            
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
                print(f"         ✅ Found {len(data)} teams")
            else:
                print(f"         ❌ Error: {response.status_code}")
            
            time.sleep(0.5)
            
    except Exception as e:
        print(f"      ❌ Error fetching talent data: {e}")
    
    return pd.DataFrame(talent_data)

def fetch_team_info_data(api_key, headers):
    """Fetch team info data (similar to cfbd_team_info())."""
    
    team_info_data = []
    
    try:
        print(f"      🏟️  Fetching team info data...")
        
        url = f'https://api.collegefootballdata.com/teams'
        params = {'year': 2024}  # Get current team info
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            for team in data:
                team_info_data.append({
                    'team_id': team.get('id'),
                    'school': team.get('school'),
                    'mascot': team.get('mascot'),
                    'abbreviation': team.get('abbreviation'),
                    'conference': team.get('conference'),
                    'classification': team.get('classification'),
                    'color': team.get('color'),
                    'alt_color': team.get('alt_color'),
                    'venue_id': team.get('venue_id'),
                    'venue_name': team.get('venue_name'),
                    'city': team.get('city'),
                    'state': team.get('state'),
                    'timezone': team.get('timezone'),
                    'latitude': team.get('latitude'),
                    'longitude': team.get('longitude'),
                    'elevation': team.get('elevation'),
                    'capacity': team.get('capacity'),
                    'year_constructed': team.get('year_constructed'),
                    'grass': team.get('grass'),
                    'dome': team.get('dome')
                })
            print(f"         ✅ Found {len(data)} teams")
        else:
            print(f"         ❌ Error: {response.status_code}")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"      ❌ Error fetching team info: {e}")
    
    return pd.DataFrame(team_info_data)

def fetch_team_matchup_data(api_key, headers, tennessee_df):
    """Fetch team matchup history data."""
    
    matchup_data = []
    
    # Get unique opponents
    opponents = set()
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        opponents.add(opponent)
    
    print(f"      📈 Fetching matchup history for {len(opponents)} opponents...")
    
    for opponent in opponents:
        print(f"         📊 {opponent} vs Tennessee...", end=" ")
        
        try:
            # Fetch matchup records
            url = f'https://api.collegefootballdata.com/teams/matchup'
            params = {'team1': 'Tennessee', 'team2': opponent, 'minYear': 2000}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    matchup_record = data[0]
                    matchup_data.append({
                        'opponent': opponent,
                        'start_year': matchup_record.get('startYear'),
                        'end_year': matchup_record.get('endYear'),
                        'tennessee_wins': matchup_record.get('team1Wins'),
                        'opponent_wins': matchup_record.get('team2Wins'),
                        'ties': matchup_record.get('ties'),
                        'total_games': matchup_record.get('team1Wins', 0) + matchup_record.get('team2Wins', 0) + matchup_record.get('ties', 0),
                        'tennessee_win_pct': matchup_record.get('team1Wins', 0) / max(1, matchup_record.get('team1Wins', 0) + matchup_record.get('team2Wins', 0))
                    })
                    print(f"✅ {matchup_record.get('team1Wins', 0)}-{matchup_record.get('team2Wins', 0)}-{matchup_record.get('ties', 0)}")
                else:
                    print(f"⚠️  No history")
            else:
                print(f"❌ {response.status_code}")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ {str(e)[:30]}...")
    
    return pd.DataFrame(matchup_data)

def create_comprehensive_dataset(tennessee_df, talent_data, team_info_data, matchup_data):
    """Create comprehensive enhanced dataset."""
    
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
        
        # Get opponent team info
        opponent_info = team_info_data[team_info_data['school'] == opponent]
        
        # Get matchup history
        matchup_history = matchup_data[matchup_data['opponent'] == opponent]
        
        # Create enhanced record
        enhanced_record = game.to_dict()
        
        # Add talent data
        enhanced_record['opponent_talent'] = talent_score
        enhanced_record['opponent_talent_tier'] = get_talent_tier(talent_score)
        enhanced_record['talent_difference'] = get_tennessee_talent(season) - talent_score
        
        # Add team info data
        if len(opponent_info) > 0:
            info = opponent_info.iloc[0]
            enhanced_record['opponent_conference'] = info.get('conference', 'Unknown')
            enhanced_record['opponent_classification'] = info.get('classification', 'FBS')
            enhanced_record['opponent_capacity'] = info.get('capacity', 50000)
            enhanced_record['opponent_grass'] = info.get('grass', True)
            enhanced_record['opponent_dome'] = info.get('dome', False)
            enhanced_record['opponent_latitude'] = info.get('latitude', 0)
            enhanced_record['opponent_longitude'] = info.get('longitude', 0)
            enhanced_record['opponent_elevation'] = info.get('elevation', 0)
            enhanced_record['opponent_year_constructed'] = info.get('year_constructed', 2000)
        else:
            # Default values
            enhanced_record['opponent_conference'] = 'Unknown'
            enhanced_record['opponent_classification'] = 'FBS'
            enhanced_record['opponent_capacity'] = 50000
            enhanced_record['opponent_grass'] = True
            enhanced_record['opponent_dome'] = False
            enhanced_record['opponent_latitude'] = 0
            enhanced_record['opponent_longitude'] = 0
            enhanced_record['opponent_elevation'] = 0
            enhanced_record['opponent_year_constructed'] = 2000
        
        # Add matchup history data
        if len(matchup_history) > 0:
            history = matchup_history.iloc[0]
            enhanced_record['historical_tennessee_wins'] = history.get('tennessee_wins', 0)
            enhanced_record['historical_opponent_wins'] = history.get('opponent_wins', 0)
            enhanced_record['historical_ties'] = history.get('ties', 0)
            enhanced_record['historical_total_games'] = history.get('total_games', 0)
            enhanced_record['historical_tennessee_win_pct'] = history.get('tennessee_win_pct', 0.5)
        else:
            # Default values
            enhanced_record['historical_tennessee_wins'] = 0
            enhanced_record['historical_opponent_wins'] = 0
            enhanced_record['historical_ties'] = 0
            enhanced_record['historical_total_games'] = 0
            enhanced_record['historical_tennessee_win_pct'] = 0.5
        
        # Add derived features
        enhanced_record['is_sec_opponent'] = enhanced_record['opponent_conference'] == 'SEC'
        enhanced_record['is_power5_opponent'] = enhanced_record['opponent_conference'] in ['SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12']
        enhanced_record['is_fcs_opponent'] = enhanced_record['opponent_classification'] == 'FCS'
        enhanced_record['venue_age'] = season - enhanced_record['opponent_year_constructed']
        enhanced_record['has_historical_rivalry'] = enhanced_record['historical_total_games'] >= 10
        
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
    tennessee_talent = {
        2022: 850,
        2023: 860,
        2024: 870
    }
    return tennessee_talent.get(season, 850)

def analyze_comprehensive_features(enhanced_df):
    """Analyze the comprehensive features."""
    
    print(f"\n📊 Comprehensive Features Analysis:")
    print("-" * 50)
    
    # Analyze by conference
    print(f"🏆 Tennessee Performance by Opponent Conference:")
    
    conf_performance = enhanced_df.groupby('opponent_conference').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean',
        'opponent_talent': 'mean'
    })
    
    for conf, stats in conf_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        avg_talent = stats[('opponent_talent', 'mean')]
        
        print(f"   vs {conf}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f} | Avg talent: {avg_talent:.0f}")
    
    # Analyze by historical rivalry
    print(f"\n📈 Tennessee Performance by Historical Rivalry:")
    
    rivalry_performance = enhanced_df.groupby('has_historical_rivalry').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean',
        'historical_tennessee_win_pct': 'mean'
    })
    
    for rivalry, stats in rivalry_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        historical_pct = stats[('historical_tennessee_win_pct', 'mean')]
        
        rivalry_name = "Rivalry" if rivalry else "Non-Rivalry"
        print(f"   {rivalry_name}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f} | Historical: {historical_pct:.1%}")
    
    # Analyze by venue characteristics
    print(f"\n🏟️  Tennessee Performance by Venue Characteristics:")
    
    # Grass vs artificial turf
    grass_performance = enhanced_df.groupby('opponent_grass').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    for grass, stats in grass_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        
        surface = "Grass" if grass else "Artificial"
        print(f"   {surface}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")
    
    # Dome vs outdoor
    dome_performance = enhanced_df.groupby('opponent_dome').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    for dome, stats in dome_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        
        venue_type = "Dome" if dome else "Outdoor"
        print(f"   {venue_type}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")
    
    # Analyze by Power 5 vs non-Power 5
    print(f"\n⚡ Tennessee Performance by Power 5 Status:")
    
    p5_performance = enhanced_df.groupby('is_power5_opponent').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean',
        'opponent_talent': 'mean'
    })
    
    for p5, stats in p5_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        avg_talent = stats[('opponent_talent', 'mean')]
        
        p5_name = "Power 5" if p5 else "Non-Power 5"
        print(f"   {p5_name}: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f} | Avg talent: {avg_talent:.0f}")

def build_comprehensive_ml_model(enhanced_df):
    """Build comprehensive ML model with all features."""
    
    print(f"\n🤖 Building Comprehensive ML Model:")
    print("-" * 50)
    
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    
    # Prepare comprehensive features
    base_features = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    # Add talent features
    talent_features = ['opponent_talent', 'talent_difference']
    
    # Add team info features
    team_info_features = [
        'opponent_capacity', 'opponent_grass', 'opponent_dome', 'opponent_latitude',
        'opponent_longitude', 'opponent_elevation', 'venue_age'
    ]
    
    # Add conference features
    conference_features = ['is_sec_opponent', 'is_power5_opponent', 'is_fcs_opponent']
    
    # Add historical features
    historical_features = [
        'historical_tennessee_wins', 'historical_opponent_wins', 'historical_ties',
        'historical_total_games', 'historical_tennessee_win_pct', 'has_historical_rivalry'
    ]
    
    comprehensive_features = base_features + talent_features + team_info_features + conference_features + historical_features
    
    # Filter to available features
    available_features = [f for f in comprehensive_features if f in enhanced_df.columns]
    
    print(f"📊 Using {len(available_features)} comprehensive features for ML")
    
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
    
    # Train comprehensive regression model
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    
    # Train comprehensive classification model
    clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_model.fit(X_train, y_train_clf)
    
    # Make predictions
    reg_predictions = reg_model.predict(X_test)
    clf_predictions = clf_model.predict(X_test)
    clf_probabilities = clf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate comprehensive models
    reg_mse = mean_squared_error(y_test_reg, reg_predictions)
    reg_r2 = r2_score(y_test_reg, reg_predictions)
    reg_mae = np.mean(np.abs(y_test_reg - reg_predictions))
    
    clf_accuracy = accuracy_score(y_test_clf, clf_predictions)
    
    print(f"Comprehensive Model Performance:")
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
    
    # Compare with previous models
    print(f"\n📊 Model Comparison:")
    print(f"   Original R²: 0.760")
    print(f"   Talent R²: 0.543")
    print(f"   Comprehensive R²: {reg_r2:.3f}")
    print(f"   Improvement over Talent: {reg_r2 - 0.543:+.3f}")
    
    print(f"   Original Accuracy: 1.000")
    print(f"   Talent Accuracy: 0.846")
    print(f"   Comprehensive Accuracy: {clf_accuracy:.3f}")
    print(f"   Change from Talent: {clf_accuracy - 0.846:+.3f}")
    
    # Show 2024 predictions
    print(f"\n🎯 2024 Predictions with Comprehensive Model:")
    print("-" * 50)
    
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        talent_tier = game['opponent_talent_tier']
        talent_score = game['opponent_talent']
        conference = game['opponent_conference']
        historical_pct = game['historical_tennessee_win_pct']
        
        pred_diff = reg_predictions[i]
        pred_win = clf_predictions[i]
        pred_prob = clf_probabilities[i]
        
        actual_diff = game['tennessee_point_differential']
        actual_win = game['tennessee_won']
        
        home_away = "vs" if game['is_home_game'] else "@"
        win_indicator = "✅" if pred_win else "❌"
        actual_indicator = "✅" if actual_win else "❌"
        
        print(f"   Week {game['week']}: {home_away} {opponent}")
        print(f"      ({talent_tier}, {talent_score:.0f}) | {conference} | Historical: {historical_pct:.1%}")
        print(f"      Predicted: {win_indicator} {pred_diff:+.1f} points (Win prob: {pred_prob:.1%})")
        print(f"      Actual:    {actual_indicator} {actual_diff:+.1f} points")
        print(f"      Error:     {abs(pred_diff - actual_diff):.1f} points")
        print()
    
    # Model validation insights
    print(f"\n🔍 Comprehensive Model Insights:")
    print("-" * 50)
    
    # Analyze prediction accuracy by conference
    test_df['predicted_win'] = clf_predictions
    test_df['prediction_error'] = np.abs(reg_predictions - y_test_reg)
    test_df['win_prediction_correct'] = clf_predictions == y_test_clf
    
    conf_accuracy = test_df.groupby('opponent_conference')['win_prediction_correct'].agg(['count', 'sum', 'mean'])
    print(f"Prediction accuracy by conference:")
    for conf, stats in conf_accuracy.iterrows():
        if stats['count'] > 0:
            print(f"   {conf}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Analyze prediction accuracy by historical rivalry
    rivalry_accuracy = test_df.groupby('has_historical_rivalry')['win_prediction_correct'].agg(['count', 'sum', 'mean'])
    print(f"\nPrediction accuracy by historical rivalry:")
    for rivalry, stats in rivalry_accuracy.iterrows():
        if stats['count'] > 0:
            rivalry_name = "Rivalry" if rivalry else "Non-Rivalry"
            print(f"   {rivalry_name}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")

if __name__ == "__main__":
    main()
