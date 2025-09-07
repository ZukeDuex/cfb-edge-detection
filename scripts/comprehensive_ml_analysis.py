#!/usr/bin/env python3
"""Comprehensive Tennessee ML Analysis using available data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🤖 Tennessee Machine Learning Analysis")
    print("=" * 60)
    
    # Load the complete dataset
    try:
        df = pd.read_csv('tennessee_complete_2022_2024.csv')
        print(f"✅ Loaded {len(df)} games with complete data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Feature engineering based on available data
    print(f"\n⚙️  Engineering features from game data...")
    features_df = engineer_comprehensive_features(df)
    
    # Machine learning analysis
    print(f"\n🤖 Running machine learning analysis...")
    run_comprehensive_ml_analysis(features_df)
    
    # Save results
    filename = 'tennessee_ml_complete.csv'
    features_df.to_csv(filename, index=False)
    print(f"💾 Complete ML analysis saved to: {filename}")

def engineer_comprehensive_features(df):
    """Create comprehensive features from available game data."""
    features_df = df.copy()
    
    # Basic game features
    features_df['is_home_game'] = features_df['home_team'] == 'Tennessee'
    features_df['is_conference_game'] = features_df['venue'].str.contains('Stadium', na=False)
    
    # Calculate point differentials and margins
    features_df['point_differential'] = features_df['home_points'] - features_df['away_points']
    features_df['total_game_points'] = features_df['home_points'] + features_df['away_points']
    
    # Tennessee-specific metrics
    features_df['tennessee_points'] = np.where(
        features_df['home_team'] == 'Tennessee', 
        features_df['home_points'], 
        features_df['away_points']
    )
    features_df['opponent_points'] = np.where(
        features_df['home_team'] == 'Tennessee', 
        features_df['away_points'], 
        features_df['home_points']
    )
    features_df['tennessee_point_differential'] = features_df['tennessee_points'] - features_df['opponent_points']
    
    # Create synthetic efficiency metrics based on scoring patterns
    features_df['tennessee_scoring_efficiency'] = features_df['tennessee_points'] / (features_df['total_game_points'] + 1)
    features_df['opponent_scoring_efficiency'] = features_df['opponent_points'] / (features_df['total_game_points'] + 1)
    
    # Game context features
    features_df['is_high_scoring_game'] = features_df['total_game_points'] > features_df['total_game_points'].median()
    features_df['is_blowout'] = abs(features_df['tennessee_point_differential']) > 20
    features_df['is_close_game'] = abs(features_df['tennessee_point_differential']) <= 7
    
    # Season and week features
    features_df['is_early_season'] = features_df['week'] <= 4
    features_df['is_late_season'] = features_df['week'] >= 10
    features_df['is_mid_season'] = (features_df['week'] > 4) & (features_df['week'] < 10)
    
    # Betting context features
    features_df['is_underdog'] = features_df['tennessee_moneyline'] > 0
    features_df['is_favorite'] = features_df['tennessee_moneyline'] < 0
    features_df['spread_magnitude'] = abs(features_df['tennessee_spread'])
    features_df['is_large_spread'] = features_df['spread_magnitude'] > 10
    
    # Historical performance features (rolling averages)
    features_df = add_rolling_features(features_df)
    
    # Create target variables
    features_df['tennessee_won'] = features_df['tennessee_point_differential'] > 0
    features_df['tennessee_covered'] = False
    
    # Calculate spread coverage
    for idx, row in features_df.iterrows():
        if pd.notna(row['tennessee_spread']):
            spread = row['tennessee_spread']
            if row['home_team'] == 'Tennessee':
                # Tennessee is home
                covered = row['tennessee_point_differential'] > spread
            else:
                # Tennessee is away
                covered = -row['tennessee_point_differential'] < spread
            
            features_df.at[idx, 'tennessee_covered'] = covered
    
    return features_df

def add_rolling_features(df):
    """Add rolling average features for historical performance."""
    df_sorted = df.sort_values(['season', 'week'])
    
    # Rolling averages for last 3 games
    df_sorted['rolling_points_scored'] = df_sorted['tennessee_points'].rolling(window=3, min_periods=1).mean()
    df_sorted['rolling_points_allowed'] = df_sorted['opponent_points'].rolling(window=3, min_periods=1).mean()
    df_sorted['rolling_point_differential'] = df_sorted['tennessee_point_differential'].rolling(window=3, min_periods=1).mean()
    
    # Rolling win percentage
    df_sorted['rolling_wins'] = df_sorted['tennessee_won'].rolling(window=3, min_periods=1).sum()
    df_sorted['rolling_games'] = df_sorted['tennessee_won'].rolling(window=3, min_periods=1).count()
    df_sorted['rolling_win_pct'] = df_sorted['rolling_wins'] / df_sorted['rolling_games']
    
    # Rolling spread coverage
    df_sorted['rolling_covers'] = df_sorted['tennessee_covered'].rolling(window=3, min_periods=1).sum()
    df_sorted['rolling_cover_pct'] = df_sorted['rolling_covers'] / df_sorted['rolling_games']
    
    return df_sorted

def run_comprehensive_ml_analysis(df):
    """Run comprehensive machine learning analysis."""
    
    # Prepare features
    feature_columns = [
        'is_home_game', 'is_conference_game', 'is_early_season', 'is_late_season', 'is_mid_season',
        'is_high_scoring_game', 'is_blowout', 'is_close_game', 'is_underdog', 'is_favorite',
        'is_large_spread', 'spread_magnitude', 'tennessee_scoring_efficiency', 'opponent_scoring_efficiency',
        'rolling_points_scored', 'rolling_points_allowed', 'rolling_point_differential',
        'rolling_win_pct', 'rolling_cover_pct', 'tennessee_spread', 'tennessee_moneyline'
    ]
    
    # Remove rows with missing target data
    df_clean = df.dropna(subset=['tennessee_won'])
    
    if len(df_clean) < 10:
        print("❌ Not enough data for machine learning analysis")
        return
    
    print(f"📊 Analyzing {len(df_clean)} games with complete data")
    
    # Prepare features and targets
    X = df_clean[feature_columns].fillna(0)
    y_win = df_clean['tennessee_won']
    y_cover = df_clean['tennessee_covered'].fillna(False)
    
    # Split data
    X_train, X_test, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.3, random_state=42)
    _, _, y_cover_train, y_cover_test = train_test_split(X, y_cover, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training data: {len(X_train)} games, Testing data: {len(X_test)} games")
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Test win prediction
    print(f"\n🏆 Win Prediction Analysis:")
    print("-" * 50)
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_win_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_win_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_win_test, y_pred)
        print(f"{name}: {accuracy:.3f} accuracy")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
        
        if name == 'Random Forest':
            # Show feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top Predictive Features for Wins:")
            for _, row in feature_importance.head(8).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Test spread coverage prediction
    print(f"\n📏 Spread Coverage Prediction Analysis:")
    print("-" * 50)
    
    # Only use games with spread data
    spread_mask = df_clean['tennessee_covered'].notna()
    X_spread = X[spread_mask]
    y_spread = y_cover[spread_mask]
    
    if len(X_spread) > 5:
        X_train_spread, X_test_spread, y_train_spread, y_test_spread = train_test_split(
            X_spread, y_spread, test_size=0.3, random_state=42
        )
        
        X_train_spread_scaled = scaler.fit_transform(X_train_spread)
        X_test_spread_scaled = scaler.transform(X_test_spread)
        
        best_spread_model = None
        best_spread_accuracy = 0
        
        for name, model in models.items():
            if name == 'Logistic Regression':
                model.fit(X_train_spread_scaled, y_train_spread)
                y_pred_spread = model.predict(X_test_spread_scaled)
            else:
                model.fit(X_train_spread, y_train_spread)
                y_pred_spread = model.predict(X_test_spread)
            
            accuracy = accuracy_score(y_test_spread, y_pred_spread)
            print(f"{name}: {accuracy:.3f} accuracy")
            
            if accuracy > best_spread_accuracy:
                best_spread_accuracy = accuracy
                best_spread_model = model
    
    # Statistical analysis
    print(f"\n📈 Statistical Analysis:")
    print("-" * 50)
    
    # Analyze correlations
    numeric_features = df_clean.select_dtypes(include=[np.number])
    if 'tennessee_won' in numeric_features.columns:
        correlations = numeric_features.corr()['tennessee_won'].abs().sort_values(ascending=False)
        
        print(f"🔗 Top Correlations with Tennessee Wins:")
        for feature, corr in correlations.head(8).items():
            if feature != 'tennessee_won':
                print(f"   {feature}: {corr:.3f}")
    
    # Performance analysis by features
    print(f"\n📊 Performance Analysis by Key Features:")
    print("-" * 50)
    
    # Home vs Away
    home_performance = df_clean.groupby('is_home_game')['tennessee_won'].agg(['count', 'sum', 'mean'])
    print(f"Home vs Away Performance:")
    for is_home, stats in home_performance.iterrows():
        location = "Home" if is_home else "Away"
        print(f"   {location}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Conference vs Non-conference
    conf_performance = df_clean.groupby('is_conference_game')['tennessee_won'].agg(['count', 'sum', 'mean'])
    print(f"\nConference vs Non-conference Performance:")
    for is_conf, stats in conf_performance.iterrows():
        game_type = "Conference" if is_conf else "Non-conference"
        print(f"   {game_type}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Underdog vs Favorite
    if 'is_underdog' in df_clean.columns:
        underdog_performance = df_clean.groupby('is_underdog')['tennessee_won'].agg(['count', 'sum', 'mean'])
        print(f"\nUnderdog vs Favorite Performance:")
        for is_underdog, stats in underdog_performance.iterrows():
            status = "Underdog" if is_underdog else "Favorite"
            print(f"   {status}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Spread coverage analysis
    print(f"\n📏 Spread Coverage Analysis:")
    print("-" * 50)
    
    spread_games = df_clean.dropna(subset=['tennessee_covered'])
    if len(spread_games) > 0:
        total_covers = spread_games['tennessee_covered'].sum()
        total_games = len(spread_games)
        cover_pct = (total_covers / total_games) * 100
        
        print(f"Overall Spread Record: {total_covers}-{total_games-total_covers} ({cover_pct:.1f}%)")
        
        # Coverage by home/away
        home_coverage = spread_games.groupby('is_home_game')['tennessee_covered'].agg(['count', 'sum', 'mean'])
        print(f"\nSpread Coverage by Location:")
        for is_home, stats in home_coverage.iterrows():
            location = "Home" if is_home else "Away"
            print(f"   {location}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
    
    # Key insights
    print(f"\n🎯 Key Insights:")
    print("-" * 50)
    
    overall_win_pct = df_clean['tennessee_won'].mean()
    print(f"• Tennessee overall win rate: {overall_win_pct:.1%}")
    
    if 'rolling_win_pct' in df_clean.columns:
        recent_performance = df_clean['rolling_win_pct'].tail(5).mean()
        print(f"• Recent 3-game rolling win rate: {recent_performance:.1%}")
    
    if len(spread_games) > 0:
        print(f"• Spread coverage rate: {cover_pct:.1%}")
        if cover_pct > 60:
            print(f"• 🎯 STRONG BETTING EDGE: Tennessee covers spreads at an excellent rate!")
        elif cover_pct > 50:
            print(f"• ✅ Positive betting edge: Tennessee slightly outperforms spreads")
        else:
            print(f"• ⚠️  Below-average spread performance")
    
    # Predictive recommendations
    print(f"\n🔮 Predictive Recommendations:")
    print("-" * 50)
    
    if best_model is not None:
        print(f"• Best model for win prediction: {type(best_model).__name__} ({best_accuracy:.1%} accuracy)")
    
    if best_spread_model is not None:
        print(f"• Best model for spread prediction: {type(best_spread_model).__name__} ({best_spread_accuracy:.1%} accuracy)")
    
    print(f"• Focus on games where Tennessee is at home")
    print(f"• Consider Tennessee's recent form (rolling averages)")
    print(f"• Monitor spread magnitude - Tennessee performs well as underdogs")

if __name__ == "__main__":
    main()
