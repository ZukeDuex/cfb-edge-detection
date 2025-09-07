import pandas as pd

# Load the complete dataset
df = pd.read_csv('tennessee_complete_2022_2024.csv')

print('🎰 Tennessee Betting Data Summary (2022-2024)')
print('=' * 60)
print(f'Total games analyzed: {len(df)}')
print()

# Performance by season
print('📊 Performance by Season:')
for season in sorted(df['season'].unique()):
    season_data = df[df['season'] == season]
    wins = season_data['tennessee_won'].sum()
    total = len(season_data)
    covers = season_data['tennessee_covered'].sum()
    total_spreads = len(season_data.dropna(subset=['tennessee_covered']))
    
    win_pct = (wins / total) * 100 if total > 0 else 0
    cover_pct = (covers / total_spreads) * 100 if total_spreads > 0 else 0
    
    print(f'   {season}: {wins}-{total-wins} ({win_pct:.1f}%) | Spread: {covers}-{total_spreads-covers} ({cover_pct:.1f}%)')

print()
print('🏆 Recent Games with Betting Data:')
recent = df.tail(10)
for _, game in recent.iterrows():
    home_team = game['home_team']
    away_team = game['away_team']
    week = game['week']
    season = game['season']
    home_pts = int(game['home_points']) if pd.notna(game['home_points']) else 'TBD'
    away_pts = int(game['away_points']) if pd.notna(game['away_points']) else 'TBD'
    
    spread = game['tennessee_spread'] if pd.notna(game['tennessee_spread']) else 'N/A'
    ml = game['tennessee_moneyline'] if pd.notna(game['tennessee_moneyline']) else 'N/A'
    
    won = '✅' if game['tennessee_won'] else '❌'
    covered = '✅' if game['tennessee_covered'] else '❌'
    
    print(f'   {won} {season} Week {week}: {away_team} @ {home_team} ({away_pts}-{home_pts}) | Spread: {spread} {covered} | ML: {ml}')

print()
print('📈 Overall Betting Performance:')
total_wins = df['tennessee_won'].sum()
total_games = len(df)
total_covers = df['tennessee_covered'].sum()
total_spread_games = len(df.dropna(subset=['tennessee_covered']))

win_pct = (total_wins / total_games) * 100
cover_pct = (total_covers / total_spread_games) * 100

print(f'   Win-Loss: {total_wins}-{total_games-total_wins} ({win_pct:.1f}%)')
print(f'   Against Spread: {total_covers}-{total_spread_games-total_covers} ({cover_pct:.1f}%)')
print(f'   Spread Performance: {"Excellent" if cover_pct > 60 else "Good" if cover_pct > 50 else "Below Average"}')

print()
print('📁 Files Created:')
print('   - tennessee_games_2022_2024.csv: Game results and details')
print('   - tennessee_odds_2022_2024.csv: Betting odds data')
print('   - tennessee_complete_2022_2024.csv: Combined dataset')
