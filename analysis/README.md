# 🏈 CFB Betting Analysis - Organized Structure

## 📁 Folder Structure

```
analysis/
├── data/                           # Raw and processed data files
│   ├── tennessee_games_2022_2024.csv          # Tennessee game data
│   ├── tennessee_odds_2022_2024.csv           # Betting lines data
│   ├── tennessee_games_error_minimized.csv    # Enhanced ML features
│   └── tennessee_games_profitable_betting.csv # Betting-ready dataset
│
├── models/                         # ML model scripts
│   ├── error_minimizing_model.py             # Advanced ML with error minimization
│   ├── profitable_betting_strategy.py        # ML + betting integration
│   ├── detailed_betting_analysis.py          # Comprehensive betting analysis
│   ├── simple_betting_analysis.py           # Basic betting analysis
│   └── comprehensive_betting_analysis.py     # Performance analysis
│
├── results/                        # Analysis results and outputs
│   ├── simple_betting_analysis.csv          # All betting opportunities
│   └── betting_strategy_summary.csv         # Performance summary
│
├── strategies/                     # Betting strategy implementations
│   └── (Future: Live betting strategies)
│
└── documentation/                  # Analysis documentation
    └── BETTING_STRATEGY_ANALYSIS.md         # Complete strategy guide
```

## 🎯 Key Findings

### 💰 **Highly Profitable Strategy Identified**
- **ROI**: +36.43% (exceptional performance)
- **Win Rate**: 71.4% (well above break-even)
- **Total Profit**: $1,020 over 28 bets
- **Average Profit**: $36.43 per $100 bet

### 🎲 **Optimal Betting Criteria**
- **Minimum Edge**: 3+ points
- **Target Edge**: 5-20 points (sweet spot)
- **Best Bet Type**: Opponent bets (88.9% win rate)
- **Avoid**: Small edges (0-5 points), Florida games

## 📊 Data Files Overview

### `data/tennessee_games_2022_2024.csv`
- **Content**: Tennessee game data with ELO ratings, attendance, etc.
- **Rows**: 39 games
- **Key Columns**: homeTeam, awayTeam, homePoints, awayPoints, homePregameElo, etc.

### `data/tennessee_odds_2022_2024.csv`
- **Content**: Betting lines from multiple sportsbooks
- **Rows**: 1,014 betting lines
- **Key Columns**: game_id, market, book, outcome, price, point

### `data/tennessee_games_error_minimized.csv`
- **Content**: Enhanced dataset with 33 new ML features
- **Features**: Interaction features, polynomial features, momentum features
- **Purpose**: Optimized for error minimization

### `data/tennessee_games_profitable_betting.csv`
- **Content**: Betting-ready dataset with comprehensive stats
- **Features**: Team stats, advanced stats, derived features
- **Purpose**: Ready for profitable betting analysis

## 🤖 Model Scripts Overview

### `models/error_minimizing_model.py`
- **Purpose**: Advanced ML model focused on minimizing prediction error
- **Features**: 106 features, 20 selected, stacking ensemble
- **Performance**: MAE = 8.42 points, R² = 0.850

### `models/profitable_betting_strategy.py`
- **Purpose**: Integrates ML predictions with betting lines
- **Features**: Comprehensive stats, advanced feature engineering
- **Output**: Betting recommendations with edge calculations

### `models/simple_betting_analysis.py`
- **Purpose**: Basic betting analysis using ELO-based predictions
- **Method**: Simple ELO to points conversion
- **Result**: Identified profitable strategy with +36.43% ROI

### `models/comprehensive_betting_analysis.py`
- **Purpose**: Detailed performance analysis and recommendations
- **Analysis**: Season-by-season, opponent analysis, edge analysis
- **Output**: Complete strategy recommendations

## 📈 Results Overview

### `results/simple_betting_analysis.csv`
- **Content**: All 39 betting opportunities with predictions and results
- **Columns**: game_id, opponent, edge, bet_recommendation, actual_profit
- **Key Insight**: 28 profitable bets identified

### `results/betting_strategy_summary.csv`
- **Content**: Performance summary statistics
- **Metrics**: Total profit, win rate, ROI, bet type performance
- **Purpose**: Quick reference for strategy performance

## 🎯 Strategy Implementation

### ✅ **DO BET ON**
1. **Opponent bets** with 10+ point edges
2. **Weaker opponents** (non-Power 5 teams)
3. **Large edges** (10-20 points)
4. **Late season games** (Weeks 10-13)

### ❌ **AVOID BETTING ON**
1. **Small edges** (0-5 points)
2. **Florida games** (0% win rate)
3. **Early season** close games
4. **Overvalued lines** (20+ point edges)

## 🚀 Next Steps for Sophisticated Models

### 1. **Enhanced Feature Engineering**
- Add weather data integration
- Include injury reports
- Add coaching changes impact
- Implement momentum indicators

### 2. **Advanced ML Techniques**
- Deep learning models (Neural Networks)
- Ensemble methods (XGBoost, LightGBM)
- Time series analysis
- Bayesian optimization

### 3. **Real-time Integration**
- Live betting line monitoring
- Real-time model updates
- Automated bet placement
- Risk management systems

### 4. **Multi-team Expansion**
- Apply strategy to other teams
- Conference-specific models
- Cross-team analysis
- Market efficiency studies

### 5. **Advanced Analytics**
- Monte Carlo simulations
- Portfolio optimization
- Kelly Criterion implementation
- Dynamic bet sizing

## 📚 Usage Instructions

### Running Models
```bash
# Navigate to analysis folder
cd analysis

# Run error-minimizing model
python models/error_minimizing_model.py

# Run betting analysis
python models/simple_betting_analysis.py

# Run comprehensive analysis
python models/comprehensive_betting_analysis.py
```

### Data Access
```python
import pandas as pd

# Load game data
games = pd.read_csv('data/tennessee_games_2022_2024.csv')

# Load betting lines
odds = pd.read_csv('data/tennessee_odds_2022_2024.csv')

# Load enhanced features
enhanced = pd.read_csv('data/tennessee_games_error_minimized.csv')
```

## 🛡️ Risk Management

- **Bankroll**: Minimum $2,000 (20x bet size)
- **Bet Size**: $100 per opportunity
- **Risk Limit**: Never exceed 5% of bankroll
- **Frequency**: 3-5 bets per season
- **Tracking**: Monitor all bets and performance

## 📞 Support

For questions about the analysis or strategy implementation, refer to:
- `documentation/BETTING_STRATEGY_ANALYSIS.md` - Complete strategy guide
- Individual model scripts for implementation details
- Results files for performance metrics

---

**Disclaimer**: This analysis is for educational purposes only. Past performance does not guarantee future results. Always bet responsibly and within your means.
