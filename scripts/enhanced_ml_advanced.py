#!/usr/bin/env python3
"""Enhanced ML model with advanced techniques for better prediction accuracy."""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from app.config import settings
import requests
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Enhanced ML Model with Advanced Techniques")
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
    
    # 3. Create synthetic matchup data
    print(f"   📈 Creating synthetic matchup history...")
    matchup_data = create_synthetic_matchup_data(tennessee_df)
    
    # 4. Create comprehensive enhanced dataset
    print(f"\n🔗 Creating comprehensive enhanced dataset...")
    enhanced_df = create_comprehensive_dataset(tennessee_df, talent_data, team_info_data, matchup_data)
    
    # 5. Advanced feature engineering
    print(f"\n⚙️  Advanced feature engineering...")
    engineered_df = advanced_feature_engineering(enhanced_df)
    
    # 6. Build enhanced ML models
    print(f"\n🤖 Building enhanced ML models...")
    build_enhanced_ml_models(engineered_df)
    
    # Save enhanced data
    filename = 'tennessee_games_enhanced_ml.csv'
    engineered_df.to_csv(filename, index=False)
    print(f"💾 Enhanced data saved to: {filename}")

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
    """Fetch team info data."""
    
    team_info_data = []
    
    try:
        print(f"      🏟️  Fetching team info data...")
        
        url = f'https://api.collegefootballdata.com/teams'
        params = {'year': 2024}
        
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

def create_synthetic_matchup_data(tennessee_df):
    """Create synthetic matchup history data."""
    
    print(f"      📈 Creating synthetic matchup history...")
    
    # Get unique opponents
    opponents = set()
    for _, game in tennessee_df.iterrows():
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        opponents.add(opponent)
    
    matchup_data = []
    
    # Define realistic matchup histories based on team relationships
    matchup_histories = {
        # SEC rivals (long history)
        'Alabama': {'tennessee_wins': 15, 'opponent_wins': 45, 'ties': 2},
        'Georgia': {'tennessee_wins': 12, 'opponent_wins': 38, 'ties': 1},
        'Florida': {'tennessee_wins': 20, 'opponent_wins': 30, 'ties': 0},
        'LSU': {'tennessee_wins': 18, 'opponent_wins': 22, 'ties': 1},
        'Kentucky': {'tennessee_wins': 35, 'opponent_wins': 15, 'ties': 0},
        'South Carolina': {'tennessee_wins': 25, 'opponent_wins': 20, 'ties': 0},
        'Missouri': {'tennessee_wins': 8, 'opponent_wins': 12, 'ties': 0},
        'Arkansas': {'tennessee_wins': 15, 'opponent_wins': 10, 'ties': 0},
        'Vanderbilt': {'tennessee_wins': 40, 'opponent_wins': 10, 'ties': 0},
        'Mississippi State': {'tennessee_wins': 20, 'opponent_wins': 15, 'ties': 0},
        'Texas A&M': {'tennessee_wins': 2, 'opponent_wins': 3, 'ties': 0},
        
        # Non-SEC opponents (limited history)
        'Oklahoma': {'tennessee_wins': 3, 'opponent_wins': 2, 'ties': 0},
        'Ohio State': {'tennessee_wins': 1, 'opponent_wins': 2, 'ties': 0},
        'Clemson': {'tennessee_wins': 2, 'opponent_wins': 1, 'ties': 0},
        'Iowa': {'tennessee_wins': 1, 'opponent_wins': 1, 'ties': 0},
        'NC State': {'tennessee_wins': 2, 'opponent_wins': 1, 'ties': 0},
        'Pittsburgh': {'tennessee_wins': 1, 'opponent_wins': 1, 'ties': 0},
        'UTSA': {'tennessee_wins': 1, 'opponent_wins': 0, 'ties': 0},
        'Virginia': {'tennessee_wins': 1, 'opponent_wins': 0, 'ties': 0},
        
        # FCS opponents (no history)
        'UT Martin': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        'Austin Peay': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        'Chattanooga': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        
        # MAC opponents (no history)
        'Ball State': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        'Akron': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        'Kent State': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        'UTEP': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
        'UConn': {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0},
    }
    
    for opponent in opponents:
        history = matchup_histories.get(opponent, {'tennessee_wins': 0, 'opponent_wins': 0, 'ties': 0})
        
        total_games = history['tennessee_wins'] + history['opponent_wins'] + history['ties']
        tennessee_win_pct = history['tennessee_wins'] / max(1, total_games)
        
        matchup_data.append({
            'opponent': opponent,
            'start_year': 1900,
            'end_year': 2024,
            'tennessee_wins': history['tennessee_wins'],
            'opponent_wins': history['opponent_wins'],
            'ties': history['ties'],
            'total_games': total_games,
            'tennessee_win_pct': tennessee_win_pct
        })
    
    print(f"         ✅ Created matchup history for {len(matchup_data)} opponents")
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
        enhanced_record['venue_age'] = season - (enhanced_record['opponent_year_constructed'] or 2000)
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

def advanced_feature_engineering(df):
    """Advanced feature engineering for better ML performance."""
    
    print(f"   🔧 Creating advanced features...")
    
    engineered_df = df.copy()
    
    # 1. Interaction features
    engineered_df['talent_home_interaction'] = engineered_df['opponent_talent'] * engineered_df['is_home_game']
    engineered_df['talent_week_interaction'] = engineered_df['opponent_talent'] * engineered_df['week']
    engineered_df['historical_home_interaction'] = engineered_df['historical_tennessee_win_pct'] * engineered_df['is_home_game']
    
    # 2. Polynomial features
    engineered_df['talent_squared'] = engineered_df['opponent_talent'] ** 2
    engineered_df['week_squared'] = engineered_df['week'] ** 2
    engineered_df['attendance_squared'] = engineered_df['attendance'] ** 2
    
    # 3. Ratio features
    engineered_df['talent_ratio'] = engineered_df['opponent_talent'] / (get_tennessee_talent(engineered_df['season'].iloc[0]) + 1)
    engineered_df['historical_ratio'] = engineered_df['historical_tennessee_wins'] / (engineered_df['historical_total_games'] + 1)
    engineered_df['capacity_ratio'] = engineered_df['opponent_capacity'] / 100000  # Normalize capacity
    
    # 4. Categorical encoding
    engineered_df['opponent_tier_encoded'] = engineered_df['opponent_talent_tier'].map({
        'FCS': 1, 'Weak': 2, 'Average': 3, 'Strong': 4, 'Elite': 5
    })
    
    # 5. Time-based features
    engineered_df['is_early_season'] = engineered_df['week'] <= 4
    engineered_df['is_mid_season'] = (engineered_df['week'] > 4) & (engineered_df['week'] < 10)
    engineered_df['is_late_season'] = engineered_df['week'] >= 10
    engineered_df['is_postseason'] = engineered_df['seasonType'] == 'postseason'
    
    # 6. Momentum features (rolling averages)
    engineered_df = engineered_df.sort_values(['season', 'week'])
    engineered_df['tennessee_points_rolling'] = engineered_df.groupby('season')['tennessee_point_differential'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    engineered_df['opponent_points_rolling'] = engineered_df.groupby('season')['opponent_talent'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    
    # 7. Distance features
    # Tennessee coordinates (approximate)
    tennessee_lat, tennessee_lon = 35.9606, -83.9207
    engineered_df['distance_from_tennessee'] = ((engineered_df['opponent_latitude'] - tennessee_lat) ** 2 + 
        (engineered_df['opponent_longitude'] - tennessee_lon) ** 2) ** 0.5
    
    # 8. Weather-related features (simplified)
    engineered_df['is_cold_weather'] = engineered_df['opponent_latitude'] > 40  # Northern teams
    engineered_df['is_warm_weather'] = engineered_df['opponent_latitude'] < 30  # Southern teams
    
    # 9. Venue features
    engineered_df['is_modern_venue'] = engineered_df['venue_age'] < 20
    engineered_df['is_old_venue'] = engineered_df['venue_age'] > 50
    
    # 10. Conference strength features
    conference_strength = {
        'SEC': 0.9, 'Big Ten': 0.85, 'Big 12': 0.8, 'ACC': 0.75, 'Pac-12': 0.7,
        'American Athletic': 0.6, 'Mountain West': 0.55, 'MAC': 0.5, 'Sun Belt': 0.45,
        'Conference USA': 0.4, 'FBS Independents': 0.6, 'Big South-OVC': 0.3,
        'Southern': 0.2, 'UAC': 0.25
    }
    engineered_df['conference_strength'] = engineered_df['opponent_conference'].map(conference_strength).fillna(0.5)
    
    print(f"      ✅ Created {len(engineered_df.columns) - len(df.columns)} new features")
    
    return engineered_df

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

def build_enhanced_ml_models(df):
    """Build enhanced ML models with advanced techniques."""
    
    print(f"\n🤖 Building Enhanced ML Models:")
    print("-" * 50)
    
    # Prepare features
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index',
        'opponent_talent', 'talent_difference', 'opponent_capacity', 'opponent_grass',
        'opponent_dome', 'opponent_latitude', 'opponent_longitude', 'opponent_elevation',
        'venue_age', 'is_sec_opponent', 'is_power5_opponent', 'is_fcs_opponent',
        'historical_tennessee_wins', 'historical_opponent_wins', 'historical_ties',
        'historical_total_games', 'historical_tennessee_win_pct', 'has_historical_rivalry',
        'talent_home_interaction', 'talent_week_interaction', 'historical_home_interaction',
        'talent_squared', 'week_squared', 'attendance_squared', 'talent_ratio',
        'historical_ratio', 'capacity_ratio', 'opponent_tier_encoded',
        'tennessee_points_rolling', 'opponent_points_rolling', 'distance_from_tennessee',
        'is_cold_weather', 'is_warm_weather', 'is_modern_venue', 'is_old_venue',
        'conference_strength'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in df.columns]
    
    print(f"📊 Using {len(available_features)} features for ML")
    
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
    
    # Feature selection
    print(f"\n🔍 Feature Selection:")
    print("-" * 30)
    
    # Select top features using mutual information
    selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(available_features)))
    X_train_selected = selector.fit_transform(X_train, y_train_reg)
    X_test_selected = selector.transform(X_test)
    
    selected_features = [available_features[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} top features:")
    for feature in selected_features:
        print(f"   {feature}")
    
    # Build ensemble models
    print(f"\n🎯 Building Ensemble Models:")
    print("-" * 30)
    
    # Regression ensemble
    reg_models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=3, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Classification ensemble
    clf_models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=3, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Train individual models
    reg_results = {}
    clf_results = {}
    
    for name, model in reg_models.items():
        print(f"   Training {name} regression...")
        
        # Use selected features for some models
        if name in ['Ridge', 'Lasso', 'SVR', 'Neural Network']:
            X_train_model = X_train_selected
            X_test_model = X_test_selected
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Scale features for linear models
        if name in ['Ridge', 'Lasso', 'SVR', 'Neural Network']:
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_model)
            X_test_scaled = scaler.transform(X_test_model)
            
            model.fit(X_train_scaled, y_train_reg)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_model, y_train_reg)
            y_pred = model.predict(X_test_model)
        
        mse = mean_squared_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)
        mae = np.mean(np.abs(y_test_reg - y_pred))
        
        reg_results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'r2': r2,
            'mae': mae
        }
        
        print(f"      R²: {r2:.3f}, MAE: {mae:.2f}")
    
    for name, model in clf_models.items():
        print(f"   Training {name} classification...")
        
        # Use selected features for some models
        if name in ['Logistic Regression', 'SVC', 'Neural Network']:
            X_train_model = X_train_selected
            X_test_model = X_test_selected
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Scale features for linear models
        if name in ['Logistic Regression', 'SVC', 'Neural Network']:
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_model)
            X_test_scaled = scaler.transform(X_test_model)
            
            model.fit(X_train_scaled, y_train_clf)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_model, y_train_clf)
            y_pred = model.predict(X_test_model)
            y_prob = model.predict_proba(X_test_model)[:, 1]
        
        accuracy = accuracy_score(y_test_clf, y_pred)
        
        clf_results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'accuracy': accuracy
        }
        
        print(f"      Accuracy: {accuracy:.3f}")
    
    # Build voting ensembles
    print(f"\n🗳️  Building Voting Ensembles:")
    print("-" * 30)
    
    # Regression voting ensemble
    reg_voting = VotingRegressor([
        ('rf', reg_models['Random Forest']),
        ('gb', reg_models['Gradient Boosting']),
        ('ridge', reg_models['Ridge'])
    ])
    
    # Scale features for voting ensemble
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    reg_voting.fit(X_train_scaled, y_train_reg)
    reg_voting_pred = reg_voting.predict(X_test_scaled)
    
    reg_voting_mse = mean_squared_error(y_test_reg, reg_voting_pred)
    reg_voting_r2 = r2_score(y_test_reg, reg_voting_pred)
    reg_voting_mae = np.mean(np.abs(y_test_reg - reg_voting_pred))
    
    print(f"   Regression Voting Ensemble:")
    print(f"      R²: {reg_voting_r2:.3f}, MAE: {reg_voting_mae:.2f}")
    
    # Classification voting ensemble
    clf_voting = VotingClassifier([
        ('rf', clf_models['Random Forest']),
        ('gb', clf_models['Gradient Boosting']),
        ('lr', clf_models['Logistic Regression'])
    ], voting='soft')
    
    clf_voting.fit(X_train_scaled, y_train_clf)
    clf_voting_pred = clf_voting.predict(X_test_scaled)
    clf_voting_prob = clf_voting.predict_proba(X_test_scaled)[:, 1]
    
    clf_voting_accuracy = accuracy_score(y_test_clf, clf_voting_pred)
    
    print(f"   Classification Voting Ensemble:")
    print(f"      Accuracy: {clf_voting_accuracy:.3f}")
    
    # Cross-validation
    print(f"\n🔄 Cross-Validation Results:")
    print("-" * 30)
    
    # Cross-validate best models
    best_reg_model = max(reg_results.items(), key=lambda x: x[1]['r2'])
    best_clf_model = max(clf_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"   Best Regression Model: {best_reg_model[0]} (R²: {best_reg_model[1]['r2']:.3f})")
    print(f"   Best Classification Model: {best_clf_model[0]} (Accuracy: {best_clf_model[1]['accuracy']:.3f})")
    
    # Cross-validation scores
    cv_scores_reg = cross_val_score(best_reg_model[1]['model'], X_train, y_train_reg, cv=5, scoring='r2')
    cv_scores_clf = cross_val_score(best_clf_model[1]['model'], X_train, y_train_clf, cv=5, scoring='accuracy')
    
    print(f"   Cross-validation R²: {cv_scores_reg.mean():.3f} (+/- {cv_scores_reg.std() * 2:.3f})")
    print(f"   Cross-validation Accuracy: {cv_scores_clf.mean():.3f} (+/- {cv_scores_clf.std() * 2:.3f})")
    
    # Model comparison
    print(f"\n📊 Model Comparison:")
    print("-" * 30)
    
    print(f"   Regression Models:")
    for name, results in reg_results.items():
        print(f"      {name}: R² = {results['r2']:.3f}, MAE = {results['mae']:.2f}")
    
    print(f"   Classification Models:")
    for name, results in clf_results.items():
        print(f"      {name}: Accuracy = {results['accuracy']:.3f}")
    
    print(f"   Ensemble Models:")
    print(f"      Regression Voting: R² = {reg_voting_r2:.3f}, MAE = {reg_voting_mae:.2f}")
    print(f"      Classification Voting: Accuracy = {clf_voting_accuracy:.3f}")
    
    # Show 2024 predictions with best models
    print(f"\n🎯 2024 Predictions with Best Models:")
    print("-" * 50)
    
    best_reg_pred = best_reg_model[1]['predictions']
    best_clf_pred = best_clf_model[1]['predictions']
    best_clf_prob = best_clf_model[1]['probabilities']
    
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        talent_tier = game['opponent_talent_tier']
        talent_score = game['opponent_talent']
        
        pred_diff = best_reg_pred[i]
        pred_win = best_clf_pred[i]
        pred_prob = best_clf_prob[i]
        
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

if __name__ == "__main__":
    main()
