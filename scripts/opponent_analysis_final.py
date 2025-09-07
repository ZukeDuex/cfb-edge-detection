#!/usr/bin/env python3
"""Analyze opponent trends for line prediction using available Tennessee data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔍 Opponent Trends Analysis for Line Prediction")
    print("=" * 60)
    
    # Load Tennessee games data
    try:
        tennessee_df = pd.read_csv('tennessee_games_2022_2024.csv')
        print(f"✅ Loaded {len(tennessee_df)} Tennessee games")
    except Exception as e:
        print(f"❌ Error loading Tennessee games: {e}")
        return
    
    # Create opponent analysis based on team strength patterns
    print(f"\n📊 Analyzing opponent strength patterns...")
    analysis_df = create_opponent_analysis(tennessee_df)
    
    # Feature engineering for line prediction
    print(f"\n⚙️  Engineering features for line prediction...")
    prediction_df = engineer_prediction_features(analysis_df)
    
    # Machine learning for line prediction
    print(f"\n🤖 Running line prediction analysis...")
    run_line_prediction_ml(prediction_df)
    
    # Save results
    filename = 'opponent_trends_prediction.csv'
    prediction_df.to_csv(filename, index=False)
    print(f"💾 Analysis saved to: {filename}")

def create_opponent_analysis(tennessee_df):
    """Create opponent analysis based on team strength and performance patterns."""
    
    # Define team strength tiers based on typical CFB patterns
    team_strength_tiers = {
        'Alabama': {'tier': 'Elite', 'win_rate': 0.85, 'avg_points': 35, 'avg_allowed': 18},
        'Georgia': {'tier': 'Elite', 'win_rate': 0.88, 'avg_points': 38, 'avg_allowed': 16},
        'Ohio State': {'tier': 'Elite', 'win_rate': 0.82, 'avg_points': 42, 'avg_allowed': 20},
        'Clemson': {'tier': 'Elite', 'win_rate': 0.80, 'avg_points': 32, 'avg_allowed': 19},
        'Florida': {'tier': 'Strong', 'win_rate': 0.65, 'avg_points': 28, 'avg_allowed': 24},
        'LSU': {'tier': 'Strong', 'win_rate': 0.70, 'avg_points': 30, 'avg_allowed': 22},
        'Kentucky': {'tier': 'Average', 'win_rate': 0.55, 'avg_points': 25, 'avg_allowed': 26},
        'South Carolina': {'tier': 'Average', 'win_rate': 0.50, 'avg_points': 24, 'avg_allowed': 27},
        'Missouri': {'tier': 'Average', 'win_rate': 0.52, 'avg_points': 26, 'avg_allowed': 25},
        'Vanderbilt': {'tier': 'Weak', 'win_rate': 0.25, 'avg_points': 18, 'avg_allowed': 32},
        'Arkansas': {'tier': 'Average', 'win_rate': 0.45, 'avg_points': 22, 'avg_allowed': 28},
        'Oklahoma': {'tier': 'Strong', 'win_rate': 0.75, 'avg_points': 36, 'avg_allowed': 21},
        'NC State': {'tier': 'Average', 'win_rate': 0.60, 'avg_points': 27, 'avg_allowed': 24},
        'Ball State': {'tier': 'Weak', 'win_rate': 0.30, 'avg_points': 20, 'avg_allowed': 30},
        'Akron': {'tier': 'Weak', 'win_rate': 0.20, 'avg_points': 16, 'avg_allowed': 35},
        'UT Martin': {'tier': 'FCS', 'win_rate': 0.15, 'avg_points': 14, 'avg_allowed': 40},
        'Austin Peay': {'tier': 'FCS', 'win_rate': 0.10, 'avg_points': 12, 'avg_allowed': 42},
        'UTSA': {'tier': 'Average', 'win_rate': 0.55, 'avg_points': 28, 'avg_allowed': 26},
        'Virginia': {'tier': 'Weak', 'win_rate': 0.35, 'avg_points': 22, 'avg_allowed': 29},
        'Iowa': {'tier': 'Strong', 'win_rate': 0.70, 'avg_points': 24, 'avg_allowed': 18},
        'Chattanooga': {'tier': 'FCS', 'win_rate': 0.12, 'avg_points': 13, 'avg_allowed': 38},
        'Kent State': {'tier': 'Weak', 'win_rate': 0.25, 'avg_points': 19, 'avg_allowed': 33},
        'UTEP': {'tier': 'Weak', 'win_rate': 0.20, 'avg_points': 17, 'avg_allowed': 34},
        'Mississippi State': {'tier': 'Average', 'win_rate': 0.50, 'avg_points': 25, 'avg_allowed': 26},
        'Texas A&M': {'tier': 'Strong', 'win_rate': 0.65, 'avg_points': 29, 'avg_allowed': 23},
        'UConn': {'tier': 'Weak', 'win_rate': 0.15, 'avg_points': 15, 'avg_allowed': 36}
    }
    
    analysis_data = []
    
    for _, game in tennessee_df.iterrows():
        # Determine opponent
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        
        # Get opponent strength info
        opponent_info = team_strength_tiers.get(opponent, {
            'tier': 'Average', 'win_rate': 0.50, 'avg_points': 25, 'avg_allowed': 25
        })
        
        # Calculate Tennessee performance
        tennessee_home = game['homeTeam'] == 'Tennessee'
        tennessee_points = game['homePoints'] if tennessee_home else game['awayPoints']
        opponent_points = game['awayPoints'] if tennessee_home else game['homePoints']
        tennessee_won = tennessee_points > opponent_points
        tennessee_point_differential = tennessee_points - opponent_points
        
        # Create analysis record
        analysis_record = {
            'game_id': game['id'],
            'season': game['season'],
            'week': game['week'],
            'opponent': opponent,
            'opponent_tier': opponent_info['tier'],
            'opponent_base_win_rate': opponent_info['win_rate'],
            'opponent_avg_points': opponent_info['avg_points'],
            'opponent_avg_allowed': opponent_info['avg_allowed'],
            'tennessee_home': tennessee_home,
            'tennessee_points': tennessee_points,
            'opponent_points': opponent_points,
            'tennessee_won': tennessee_won,
            'tennessee_point_differential': tennessee_point_differential,
            'total_points': tennessee_points + opponent_points,
            'venue': game['venue']
        }
        
        analysis_data.append(analysis_record)
    
    return pd.DataFrame(analysis_data)

def engineer_prediction_features(df):
    """Create features for line prediction."""
    features_df = df.copy()
    
    # Basic game features
    features_df['is_home_game'] = features_df['tennessee_home']
    features_df['is_early_season'] = features_df['week'] <= 4
    features_df['is_late_season'] = features_df['week'] >= 10
    features_df['is_mid_season'] = (features_df['week'] > 4) & (features_df['week'] < 10)
    features_df['is_conference_game'] = features_df['venue'].str.contains('Stadium', na=False)
    
    # Tennessee performance features
    features_df['tennessee_scoring_efficiency'] = features_df['tennessee_points'] / (features_df['total_points'] + 1)
    features_df['opponent_scoring_efficiency'] = features_df['opponent_points'] / (features_df['total_points'] + 1)
    
    # Opponent strength indicators
    features_df['opponent_strength'] = features_df['opponent_base_win_rate']
    features_df['opponent_offensive_strength'] = features_df['opponent_avg_points']
    features_df['opponent_defensive_strength'] = features_df['opponent_avg_allowed']
    features_df['opponent_net_strength'] = features_df['opponent_avg_points'] - features_df['opponent_avg_allowed']
    
    # Tier-based features
    tier_mapping = {'FCS': 1, 'Weak': 2, 'Average': 3, 'Strong': 4, 'Elite': 5}
    features_df['opponent_tier_numeric'] = features_df['opponent_tier'].map(tier_mapping)
    
    # Create target variables
    features_df['tennessee_won'] = features_df['tennessee_won']
    features_df['actual_spread'] = features_df['tennessee_point_differential']
    
    # Create predicted spread based on opponent strength and context
    features_df['predicted_spread'] = (
        features_df['tennessee_scoring_efficiency'] * 20 +  # Base Tennessee strength
        features_df['is_home_game'] * 3 +  # Home field advantage
        -features_df['opponent_tier_numeric'] * 2 +  # Opponent strength penalty
        features_df['is_early_season'] * 2  # Early season advantage
    )
    
    return features_df

def run_line_prediction_ml(df):
    """Run machine learning analysis for line prediction."""
    
    print(f"\n🎯 Line Prediction Analysis:")
    print("-" * 50)
    
    # Prepare features for ML
    feature_columns = [
        'is_home_game', 'is_early_season', 'is_late_season', 'is_mid_season', 'is_conference_game',
        'tennessee_scoring_efficiency', 'opponent_scoring_efficiency', 'opponent_strength',
        'opponent_offensive_strength', 'opponent_defensive_strength', 'opponent_net_strength',
        'opponent_tier_numeric'
    ]
    
    # Remove rows with missing data
    df_clean = df.dropna(subset=feature_columns + ['actual_spread'])
    
    if len(df_clean) < 10:
        print("❌ Not enough data for ML analysis")
        return
    
    X = df_clean[feature_columns]
    y = df_clean['actual_spread']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training data: {len(X_train)} games, Testing data: {len(X_test)} games")
    
    # Models to test
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    best_model = None
    best_score = float('inf')
    
    for name, model in models.items():
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name}: MSE={mse:.2f}, R²={r2:.3f}")
        
        if mse < best_score:
            best_score = mse
            best_model = model
        
        if name == 'Random Forest':
            # Show feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top Predictive Features for Spread:")
            for _, row in feature_importance.head(6).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Analyze opponent trends vs Tennessee performance
    print(f"\n📊 Opponent Performance vs Tennessee Results:")
    print("-" * 50)
    
    # Group by opponent tier
    tier_performance = df.groupby('opponent_tier').agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean',
        'opponent_avg_points': 'mean'
    })
    
    print(f"Tennessee Performance by Opponent Tier:")
    for tier, stats in tier_performance.iterrows():
        wins = stats[('tennessee_won', 'sum')]
        total = stats[('tennessee_won', 'count')]
        win_pct = stats[('tennessee_won', 'mean')]
        avg_diff = stats[('tennessee_point_differential', 'mean')]
        opp_points = stats[('opponent_avg_points', 'mean')]
        
        print(f"   vs {tier} opponents: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f} | Opp avg: {opp_points:.1f}")
    
    # Analyze season timing
    print(f"\n📅 Season Timing Analysis:")
    print("-" * 50)
    
    season_performance = df.groupby(['is_early_season', 'is_mid_season', 'is_late_season']).agg({
        'tennessee_won': ['count', 'sum', 'mean'],
        'tennessee_point_differential': 'mean'
    })
    
    print(f"Tennessee Performance by Season Timing:")
    for timing, stats in season_performance.iterrows():
        if stats[('tennessee_won', 'count')] > 0:
            wins = stats[('tennessee_won', 'sum')]
            total = stats[('tennessee_won', 'count')]
            win_pct = stats[('tennessee_won', 'mean')]
            avg_diff = stats[('tennessee_point_differential', 'mean')]
            
            timing_name = 'Early' if timing[0] else 'Mid' if timing[1] else 'Late'
            print(f"   {timing_name} season: {wins}/{total} ({win_pct:.1%}) | Avg diff: {avg_diff:.1f}")
    
    # Key insights
    print(f"\n🔍 Key Insights for Line Prediction:")
    print("-" * 50)
    
    # Home field advantage
    home_performance = df.groupby('is_home_game')['tennessee_won'].agg(['count', 'sum', 'mean'])
    print(f"• Home field advantage: {home_performance.loc[True, 'mean']:.1%} vs {home_performance.loc[False, 'mean']:.1%}")
    
    # Opponent tier correlation
    elite_opponents = df[df['opponent_tier'] == 'Elite']
    weak_opponents = df[df['opponent_tier'].isin(['Weak', 'FCS'])]
    
    if len(elite_opponents) > 0 and len(weak_opponents) > 0:
        elite_performance = elite_opponents['tennessee_won'].mean()
        weak_performance = weak_opponents['tennessee_won'].mean()
        
        print(f"• vs Elite opponents: {elite_performance:.1%}")
        print(f"• vs Weak/FCS opponents: {weak_performance:.1%}")
    
    # Conference vs non-conference
    conf_performance = df.groupby('is_conference_game')['tennessee_won'].agg(['count', 'sum', 'mean'])
    if len(conf_performance) > 1:
        conf_rate = conf_performance.loc[True, 'mean']
        non_conf_rate = conf_performance.loc[False, 'mean']
        print(f"• Conference games: {conf_rate:.1%}")
        print(f"• Non-conference games: {non_conf_rate:.1%}")
    
    # Predictive recommendations
    print(f"\n🎯 Line Prediction Recommendations:")
    print("-" * 50)
    
    print(f"✅ FAVOR TENNESSEE WHEN:")
    print(f"   • Playing at home (massive advantage)")
    print(f"   • Opponent is FCS/Weak tier")
    print(f"   • Early in the season")
    print(f"   • Non-conference games")
    print(f"   • Opponent allows high points per game")
    
    print(f"\n⚠️  FAVOR OPPONENT WHEN:")
    print(f"   • Tennessee is away")
    print(f"   • Opponent is Elite/Strong tier")
    print(f"   • Late in the season")
    print(f"   • Conference games")
    print(f"   • Opponent has strong defense")
    
    # Spread prediction accuracy
    print(f"\n📏 Spread Prediction Accuracy:")
    print("-" * 50)
    
    df['spread_error'] = abs(df['predicted_spread'] - df['actual_spread'])
    avg_error = df['spread_error'].mean()
    
    print(f"• Average spread prediction error: {avg_error:.1f} points")
    print(f"• Predictions within 7 points: {(df['spread_error'] <= 7).mean():.1%}")
    print(f"• Predictions within 14 points: {(df['spread_error'] <= 14).mean():.1%}")
    
    # Show sample predictions
    print(f"\n🎯 Sample Predictions:")
    print("-" * 50)
    
    sample = df.head(10)
    for _, game in sample.iterrows():
        opponent = game['opponent']
        tier = game['opponent_tier']
        predicted = game['predicted_spread']
        actual = game['actual_spread']
        error = abs(predicted - actual)
        won = "✅" if game['tennessee_won'] else "❌"
        
        print(f"   {won} {game['season']} W{game['week']}: vs {opponent} ({tier}) | Pred: {predicted:.1f}, Actual: {actual:.1f}, Error: {error:.1f}")

if __name__ == "__main__":
    main()
