#!/usr/bin/env python3
"""Build ML algorithm using 2022-2023 data to predict 2024 Tennessee results."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🤖 ML Algorithm: Predicting 2024 Tennessee Results")
    print("=" * 60)
    
    # Load Tennessee games data
    try:
        tennessee_df = pd.read_csv('tennessee_games_2022_2024.csv')
        print(f"✅ Loaded {len(tennessee_df)} Tennessee games")
    except Exception as e:
        print(f"❌ Error loading Tennessee games: {e}")
        return
    
    # Prepare data for ML
    print(f"\n📊 Preparing data for machine learning...")
    ml_df = prepare_ml_data(tennessee_df)
    
    # Split data into training (2022-2023) and testing (2024)
    print(f"\n🔄 Splitting data: 2022-2023 for training, 2024 for testing...")
    train_df, test_df = split_by_season(ml_df)
    
    print(f"   Training data: {len(train_df)} games (2022-2023)")
    print(f"   Testing data: {len(test_df)} games (2024)")
    
    if len(test_df) == 0:
        print("❌ No 2024 data available for testing")
        return
    
    # Build and train models
    print(f"\n🏗️  Building and training ML models...")
    
    # Feature columns
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_base_win_rate', 'opponent_avg_points', 'opponent_avg_allowed', 'opponent_net_strength',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    # Prepare training data
    X_train = train_df[feature_columns].fillna(0)
    y_train_reg = train_df['tennessee_point_differential']
    y_train_clf = train_df['tennessee_won']
    
    # Prepare testing data
    X_test = test_df[feature_columns].fillna(0)
    y_test_reg = test_df['tennessee_point_differential']
    y_test_clf = test_df['tennessee_won']
    
    # Train regression model (point differential)
    print(f"\n📊 Training Regression Model (Point Differential):")
    print("-" * 50)
    
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    
    # Train classification model (win/loss)
    print(f"\n🎯 Training Classification Model (Win/Loss):")
    print("-" * 50)
    
    clf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_model.fit(X_train, y_train_clf)
    
    # Make predictions
    reg_predictions = reg_model.predict(X_test)
    clf_predictions = clf_model.predict(X_test)
    clf_probabilities = clf_model.predict_proba(X_test)[:, 1]
    
    # Evaluate models
    reg_mse = mean_squared_error(y_test_reg, reg_predictions)
    reg_r2 = r2_score(y_test_reg, reg_predictions)
    reg_mae = np.mean(np.abs(y_test_reg - reg_predictions))
    
    clf_accuracy = accuracy_score(y_test_clf, clf_predictions)
    
    print(f"\n📈 Model Performance:")
    print("-" * 50)
    print(f"Regression Model (Point Differential):")
    print(f"   MSE: {reg_mse:.2f}")
    print(f"   R²: {reg_r2:.3f}")
    print(f"   MAE: {reg_mae:.2f}")
    print(f"\nClassification Model (Win/Loss):")
    print(f"   Accuracy: {clf_accuracy:.3f}")
    
    # Feature importance
    print(f"\n🔍 Feature Importance Analysis:")
    print("-" * 50)
    
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': reg_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Display 2024 predictions
    print(f"\n🎯 2024 Game Predictions:")
    print("-" * 50)
    
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['opponent']
        tier = game['opponent_tier']
        week = game['week']
        is_home = game['is_home_game']
        
        pred_diff = reg_predictions[i]
        pred_win = clf_predictions[i]
        pred_prob = clf_probabilities[i]
        
        actual_diff = game['tennessee_point_differential']
        actual_win = game['tennessee_won']
        
        home_away = "vs" if is_home else "@"
        win_indicator = "✅" if pred_win else "❌"
        actual_indicator = "✅" if actual_win else "❌"
        
        print(f"   Week {week}: {home_away} {opponent} ({tier})")
        print(f"      Predicted: {win_indicator} {pred_diff:+.1f} points (Win prob: {pred_prob:.1%})")
        print(f"      Actual:    {actual_indicator} {actual_diff:+.1f} points")
        print(f"      Error:     {abs(pred_diff - actual_diff):.1f} points")
        print()
    
    # Model validation
    print(f"\n📈 Model Validation:")
    print("-" * 50)
    
    errors = np.abs(reg_predictions - y_test_reg)
    print(f"• Point Differential Prediction MAE: {reg_mae:.1f} points")
    print(f"• Win/Loss Prediction Accuracy: {clf_accuracy:.1%}")
    print(f"• Predictions within 7 points: {(errors <= 7).mean():.1%}")
    print(f"• Predictions within 14 points: {(errors <= 14).mean():.1%}")
    print(f"• Predictions within 21 points: {(errors <= 21).mean():.1%}")
    
    # Model reliability assessment
    print(f"\n🎯 Model Reliability Assessment:")
    print("-" * 50)
    
    if clf_accuracy >= 0.8:
        print("✅ Win/Loss prediction: HIGHLY RELIABLE")
    elif clf_accuracy >= 0.7:
        print("⚠️  Win/Loss prediction: MODERATELY RELIABLE")
    else:
        print("❌ Win/Loss prediction: LOW RELIABILITY")
    
    if reg_mae <= 10:
        print("✅ Point differential prediction: HIGHLY RELIABLE")
    elif reg_mae <= 15:
        print("⚠️  Point differential prediction: MODERATELY RELIABLE")
    else:
        print("❌ Point differential prediction: LOW RELIABILITY")
    
    # Create results summary
    results_df = test_df[['season', 'week', 'opponent', 'opponent_tier', 'is_home_game']].copy()
    results_df['predicted_point_differential'] = reg_predictions
    results_df['predicted_win'] = clf_predictions
    results_df['predicted_win_probability'] = clf_probabilities
    results_df['actual_point_differential'] = y_test_reg
    results_df['actual_win'] = y_test_clf
    results_df['prediction_error'] = errors
    results_df['win_prediction_correct'] = clf_predictions == y_test_clf
    
    # Save results
    filename = 'ml_predictions_2024.csv'
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Predictions saved to: {filename}")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    print("-" * 50)
    
    # Analyze prediction patterns
    correct_predictions = results_df['win_prediction_correct'].sum()
    total_predictions = len(results_df)
    
    print(f"• Correctly predicted {correct_predictions}/{total_predictions} games ({correct_predictions/total_predictions:.1%})")
    
    # Analyze by opponent tier
    tier_accuracy = results_df.groupby('opponent_tier')['win_prediction_correct'].agg(['count', 'sum', 'mean'])
    print(f"\n• Prediction accuracy by opponent tier:")
    for tier, stats in tier_accuracy.iterrows():
        if stats['count'] > 0:
            print(f"   {tier}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Analyze by home/away
    home_accuracy = results_df.groupby('is_home_game')['win_prediction_correct'].agg(['count', 'sum', 'mean'])
    print(f"\n• Prediction accuracy by location:")
    print(f"   Home: {home_accuracy.loc[True, 'sum']}/{home_accuracy.loc[True, 'count']} ({home_accuracy.loc[True, 'mean']:.1%})")
    print(f"   Away: {home_accuracy.loc[False, 'sum']}/{home_accuracy.loc[False, 'count']} ({home_accuracy.loc[False, 'mean']:.1%})")

def prepare_ml_data(df):
    """Prepare data for machine learning with comprehensive features."""
    
    # Define team strength tiers
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
    
    ml_data = []
    
    for _, game in df.iterrows():
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
        
        # Create comprehensive feature set
        features = {
            # Game context
            'season': game['season'],
            'week': game['week'],
            'is_home_game': tennessee_home,
            'is_conference_game': game['conferenceGame'],
            'is_neutral_site': game['neutralSite'],
            'attendance': game['attendance'] if pd.notna(game['attendance']) else 0,
            
            # Season timing
            'is_early_season': game['week'] <= 4,
            'is_mid_season': (game['week'] > 4) & (game['week'] < 10),
            'is_late_season': game['week'] >= 10,
            'is_postseason': game['seasonType'] == 'postseason',
            
            # Opponent strength
            'opponent_tier': opponent_info['tier'],
            'opponent_base_win_rate': opponent_info['win_rate'],
            'opponent_avg_points': opponent_info['avg_points'],
            'opponent_avg_allowed': opponent_info['avg_allowed'],
            'opponent_net_strength': opponent_info['avg_points'] - opponent_info['avg_allowed'],
            
            # Tennessee performance (targets)
            'tennessee_points': tennessee_points,
            'opponent_points': opponent_points,
            'tennessee_won': tennessee_won,
            'tennessee_point_differential': tennessee_point_differential,
            'total_points': tennessee_points + opponent_points,
            
            # Derived features
            'tennessee_scoring_efficiency': tennessee_points / (tennessee_points + opponent_points + 1),
            'opponent_scoring_efficiency': opponent_points / (tennessee_points + opponent_points + 1),
            
            # ELO ratings (if available)
            'tennessee_pregame_elo': game['homePregameElo'] if tennessee_home else game['awayPregameElo'],
            'opponent_pregame_elo': game['awayPregameElo'] if tennessee_home else game['homePregameElo'],
            'elo_difference': (game['homePregameElo'] - game['awayPregameElo']) if tennessee_home else (game['awayPregameElo'] - game['homePregameElo']),
            
            # Win probability
            'tennessee_pregame_win_prob': game['homePostgameWinProbability'] if tennessee_home else game['awayPostgameWinProbability'],
            'opponent_pregame_win_prob': game['awayPostgameWinProbability'] if tennessee_home else game['homePostgameWinProbability'],
            
            # Game excitement
            'excitement_index': game['excitementIndex'] if pd.notna(game['excitementIndex']) else 0,
            
            # Opponent info
            'opponent': opponent,
            'opponent_conference': game['awayConference'] if tennessee_home else game['homeConference'],
            'opponent_classification': game['awayClassification'] if tennessee_home else game['homeClassification']
        }
        
        ml_data.append(features)
    
    return pd.DataFrame(ml_data)

def split_by_season(df):
    """Split data into training (2022-2023) and testing (2024)."""
    train_df = df[df['season'].isin([2022, 2023])].copy()
    test_df = df[df['season'] == 2024].copy()
    
    return train_df, test_df

if __name__ == "__main__":
    main()
