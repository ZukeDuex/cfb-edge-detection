#!/usr/bin/env python3
"""Build ML algorithm using 2022-2023 data to predict 2024 Tennessee results."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
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
    
    # Regression models for point differential prediction
    regression_models = build_regression_models()
    regression_results = train_and_evaluate_regression(regression_models, train_df, test_df)
    
    # Classification models for win/loss prediction
    classification_models = build_classification_models()
    classification_results = train_and_evaluate_classification(classification_models, train_df, test_df)
    
    # Feature importance analysis
    print(f"\n🔍 Analyzing feature importance...")
    analyze_feature_importance(regression_models, train_df)
    
    # Make predictions for 2024
    print(f"\n🎯 Making predictions for 2024 games...")
    make_2024_predictions(regression_models, classification_models, test_df)
    
    # Model validation and insights
    print(f"\n📈 Model validation and insights...")
    validate_predictions(regression_results, classification_results, test_df)
    
    # Save results
    results_df = create_results_summary(test_df, regression_models, classification_models)
    filename = 'ml_predictions_2024.csv'
    results_df.to_csv(filename, index=False)
    print(f"💾 Predictions saved to: {filename}")

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
            'point_differential_ratio': tennessee_point_differential / (tennessee_points + opponent_points + 1),
            
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

def build_regression_models():
    """Build regression models for point differential prediction."""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    return models

def build_classification_models():
    """Build classification models for win/loss prediction."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    }
    
    return models

def train_and_evaluate_regression(models, train_df, test_df):
    """Train and evaluate regression models."""
    print(f"\n📊 Regression Model Performance:")
    print("-" * 50)
    
    # Feature columns for regression
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_base_win_rate', 'opponent_avg_points', 'opponent_avg_allowed', 'opponent_net_strength',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    # Prepare data
    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df['tennessee_point_differential']
    X_test = test_df[feature_columns].fillna(0)
    y_test = test_df['tennessee_point_differential']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name in ['Linear Regression', 'SVR']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        print(f"{name}:")
        print(f"   MSE: {mse:.2f}")
        print(f"   R²: {r2:.3f}")
        print(f"   MAE: {mae:.2f}")
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'scaler': scaler if name in ['Linear Regression', 'SVR'] else None
        }
    
    return results

def train_and_evaluate_classification(models, train_df, test_df):
    """Train and evaluate classification models."""
    print(f"\n🎯 Classification Model Performance:")
    print("-" * 50)
    
    # Feature columns for classification
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_base_win_rate', 'opponent_avg_points', 'opponent_avg_allowed', 'opponent_net_strength',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    # Prepare data
    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df['tennessee_won']
    X_test = test_df[feature_columns].fillna(0)
    y_test = test_df['tennessee_won']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name}:")
        print(f"   Accuracy: {accuracy:.3f}")
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob,
            'accuracy': accuracy,
            'scaler': scaler if name == 'Logistic Regression' else None
        }
    
    return results

def analyze_feature_importance(models, train_df):
    """Analyze feature importance from the best model."""
    print(f"\n🔍 Feature Importance Analysis:")
    print("-" * 50)
    
    # Use Random Forest for feature importance
    rf_model = models['Random Forest']
    
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_base_win_rate', 'opponent_avg_points', 'opponent_avg_allowed', 'opponent_net_strength',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df['tennessee_point_differential']
    
    rf_model.fit(X_train, y_train)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return importance_df

def make_2024_predictions(regression_models, classification_models, test_df):
    """Make predictions for 2024 games."""
    print(f"\n🎯 2024 Game Predictions:")
    print("-" * 50)
    
    # Use best regression model (Random Forest)
    best_reg_model = regression_models['Random Forest']['model']
    
    # Use best classification model (Random Forest)
    best_clf_model = classification_models['Random Forest']['model']
    
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_base_win_rate', 'opponent_avg_points', 'opponent_avg_allowed', 'opponent_net_strength',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    X_test = test_df[feature_columns].fillna(0)
    
    # Make predictions
    point_diff_pred = best_reg_model.predict(X_test)
    win_pred = best_clf_model.predict(X_test)
    win_prob = best_clf_model.predict_proba(X_test)[:, 1]
    
    # Display predictions
    for i, (_, game) in enumerate(test_df.iterrows()):
        opponent = game['opponent']
        tier = game['opponent_tier']
        week = game['week']
        is_home = game['is_home_game']
        
        pred_diff = point_diff_pred[i]
        pred_win = win_pred[i]
        pred_prob = win_prob[i]
        
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

def validate_predictions(regression_results, classification_results, test_df):
    """Validate predictions and provide insights."""
    print(f"\n📈 Prediction Validation:")
    print("-" * 50)
    
    # Get best models
    best_reg_model = regression_results['Random Forest']
    best_clf_model = classification_results['Random Forest']
    
    # Calculate overall accuracy
    reg_mae = best_reg_model['mae']
    clf_accuracy = best_clf_model['accuracy']
    
    print(f"• Point Differential Prediction MAE: {reg_mae:.1f} points")
    print(f"• Win/Loss Prediction Accuracy: {clf_accuracy:.1%}")
    
    # Analyze prediction errors
    predictions = best_reg_model['predictions']
    actual = test_df['tennessee_point_differential']
    errors = np.abs(predictions - actual)
    
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

def create_results_summary(test_df, regression_models, classification_models):
    """Create summary of predictions and results."""
    
    # Use best models
    best_reg_model = regression_models['Random Forest']['model']
    best_clf_model = classification_models['Random Forest']['model']
    
    feature_columns = [
        'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 'attendance',
        'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason',
        'opponent_base_win_rate', 'opponent_avg_points', 'opponent_avg_allowed', 'opponent_net_strength',
        'tennessee_pregame_elo', 'opponent_pregame_elo', 'elo_difference',
        'tennessee_pregame_win_prob', 'opponent_pregame_win_prob', 'excitement_index'
    ]
    
    X_test = test_df[feature_columns].fillna(0)
    
    # Make predictions
    point_diff_pred = best_reg_model.predict(X_test)
    win_pred = best_clf_model.predict(X_test)
    win_prob = best_clf_model.predict_proba(X_test)[:, 1]
    
    # Create results DataFrame
    results_df = test_df[['season', 'week', 'opponent', 'opponent_tier', 'is_home_game']].copy()
    results_df['predicted_point_differential'] = point_diff_pred
    results_df['predicted_win'] = win_pred
    results_df['predicted_win_probability'] = win_prob
    results_df['actual_point_differential'] = test_df['tennessee_point_differential']
    results_df['actual_win'] = test_df['tennessee_won']
    results_df['prediction_error'] = np.abs(point_diff_pred - test_df['tennessee_point_differential'])
    results_df['win_prediction_correct'] = win_pred == test_df['tennessee_won']
    
    return results_df

if __name__ == "__main__":
    main()
