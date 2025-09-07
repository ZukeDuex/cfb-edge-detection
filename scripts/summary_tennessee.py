import pandas as pd

# Read the Tennessee games data
df = pd.read_csv('tennessee_games_2022_2024.csv')

print('🏈 Tennessee Football Games Summary (2022-2024)')
print('=' * 60)
print(f'Total games: {len(df)}')
print()

# Summary by season
print('📅 Games by Season:')
season_summary = df.groupby('season').size()
for season, count in season_summary.items():
    print(f'   {season}: {count} games')

print()
print('🏆 Recent Games (Last 10):')
recent = df.tail(10)
for _, game in recent.iterrows():
    home_team = game['homeTeam']
    away_team = game['awayTeam']
    week = game['week']
    season = game['season']
    home_pts = int(game['homePoints']) if pd.notna(game['homePoints']) else 'TBD'
    away_pts = int(game['awayPoints']) if pd.notna(game['awayPoints']) else 'TBD'
    status = '✅' if game['completed'] else '⏳'
    venue = game['venue']
    
    # Determine if Tennessee is home or away
    if home_team == 'Tennessee':
        matchup = f'{away_team} @ Tennessee'
        score = f'({away_pts}-{home_pts})'
    else:
        matchup = f'Tennessee @ {home_team}'
        score = f'({away_pts}-{home_pts})'
    
    print(f'{status} {season} Week {week}: {matchup} {score} ({venue})')

print()
print('📊 Win-Loss Record by Season:')
for season in sorted(df['season'].unique()):
    season_games = df[df['season'] == season]
    wins = 0
    losses = 0
    
    for _, game in season_games.iterrows():
        if game['completed'] and pd.notna(game['homePoints']) and pd.notna(game['awayPoints']):
            if game['homeTeam'] == 'Tennessee':
                if game['homePoints'] > game['awayPoints']:
                    wins += 1
                else:
                    losses += 1
            else:  # Tennessee is away
                if game['awayPoints'] > game['homePoints']:
                    wins += 1
                else:
                    losses += 1
    
    total = wins + losses
    if total > 0:
        win_pct = (wins / total) * 100
        print(f'   {season}: {wins}-{losses} ({win_pct:.1f}%)')
