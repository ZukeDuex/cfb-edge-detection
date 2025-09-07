#!/usr/bin/env python3
"""Combine Tennessee games and betting odds data."""

import pandas as pd
import numpy as np

def main():
    print("🔗 Combining Tennessee Games and Betting Odds Data")
    print("=" * 60)
    
    # Load both datasets
    try:
        games_df = pd.read_csv('tennessee_games_2022_2024.csv')
        odds_df = pd.read_csv('tennessee_odds_2022_2024.csv')
        print(f"✅ Loaded {len(games_df)} games and {len(odds_df)} odds entries")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create a comprehensive dataset
    print(f"\n📊 Creating comprehensive dataset...")
    
    # Start with games data
    combined_df = games_df.copy()
    
    # Add betting odds data
    odds_summary = []
    
    for _, game in games_df.iterrows():
        game_id = game['id']
        game_odds = odds_df[odds_df['game_id'] == game_id]
        
        if not game_odds.empty:
            # Get the most recent odds for each market/book combination
            latest_odds = game_odds.groupby(['market', 'book']).last().reset_index()
            
            # Create odds summary for this game
            game_summary = {
                'game_id': game_id,
                'season': game['season'],
                'week': game['week'],
                'home_team': game['homeTeam'],
                'away_team': game['awayTeam'],
                'home_points': game['homePoints'],
                'away_points': game['awayPoints'],
                'completed': game['completed'],
                'venue': game['venue']
            }
            
            # Add spread data
            spreads = latest_odds[latest_odds['market'] == 'spreads']
            if not spreads.empty:
                # Find Tennessee spread
                tennessee_spread = spreads[spreads['outcome'].str.contains('Tennessee', case=False, na=False)]
                if not tennessee_spread.empty:
                    game_summary['tennessee_spread'] = tennessee_spread.iloc[0]['point']
                    game_summary['tennessee_spread_price'] = tennessee_spread.iloc[0]['price']
                    game_summary['tennessee_spread_book'] = tennessee_spread.iloc[0]['book']
            
            # Add total data
            totals = latest_odds[latest_odds['market'] == 'totals']
            if not totals.empty:
                over_total = totals[totals['outcome'].str.contains('Over', case=False, na=False)]
                if not over_total.empty:
                    game_summary['total_points'] = over_total.iloc[0]['point']
                    game_summary['over_price'] = over_total.iloc[0]['price']
                    game_summary['total_book'] = over_total.iloc[0]['book']
            
            # Add moneyline data
            moneylines = latest_odds[latest_odds['market'] == 'h2h']
            if not moneylines.empty:
                tennessee_ml = moneylines[moneylines['outcome'].str.contains('Tennessee', case=False, na=False)]
                if not tennessee_ml.empty:
                    game_summary['tennessee_moneyline'] = tennessee_ml.iloc[0]['price']
                    game_summary['ml_book'] = tennessee_ml.iloc[0]['book']
            
            odds_summary.append(game_summary)
    
    # Create final dataset
    final_df = pd.DataFrame(odds_summary)
    
    # Add calculated fields
    final_df['tennessee_won'] = False
    final_df['tennessee_covered'] = False
    final_df['total_over'] = False
    
    for idx, row in final_df.iterrows():
        if row['completed'] and pd.notna(row['home_points']) and pd.notna(row['away_points']):
            # Determine if Tennessee won
            if row['home_team'] == 'Tennessee':
                tennessee_won = row['home_points'] > row['away_points']
                tennessee_score = row['home_points']
                opponent_score = row['away_points']
            else:
                tennessee_won = row['away_points'] > row['home_points']
                tennessee_score = row['away_points']
                opponent_score = row['home_points']
            
            final_df.at[idx, 'tennessee_won'] = tennessee_won
            
            # Check if Tennessee covered the spread
            if pd.notna(row['tennessee_spread']):
                spread = row['tennessee_spread']
                if row['home_team'] == 'Tennessee':
                    # Tennessee is home, so they need to win by more than the spread
                    covered = (tennessee_score - opponent_score) > spread
                else:
                    # Tennessee is away, so they need to lose by less than the spread
                    covered = (opponent_score - tennessee_score) < spread
                
                final_df.at[idx, 'tennessee_covered'] = covered
            
            # Check if total went over
            if 'total_points' in row and pd.notna(row['total_points']):
                total_score = tennessee_score + opponent_score
                final_df.at[idx, 'total_over'] = total_score > row['total_points']
    
    # Save combined dataset
    filename = 'tennessee_complete_2022_2024.csv'
    final_df.to_csv(filename, index=False)
    
    print(f"✅ Created comprehensive dataset with {len(final_df)} games")
    print(f"💾 Data saved to: {filename}")
    
    # Show summary statistics
    print(f"\n📈 Summary Statistics:")
    
    # Games with betting data
    games_with_odds = final_df.dropna(subset=['tennessee_spread'])
    print(f"   Games with spread data: {len(games_with_odds)}")
    
    # Tennessee performance
    completed_games = final_df[final_df['completed'] == True]
    if not completed_games.empty:
        wins = completed_games['tennessee_won'].sum()
        total = len(completed_games)
        win_pct = (wins / total) * 100
        print(f"   Tennessee record: {wins}-{total-wins} ({win_pct:.1f}%)")
    
    # Spread performance
    spread_games = final_df.dropna(subset=['tennessee_covered'])
    if not spread_games.empty:
        covers = spread_games['tennessee_covered'].sum()
        total_spreads = len(spread_games)
        cover_pct = (covers / total_spreads) * 100
        print(f"   Spread record: {covers}-{total_spreads-covers} ({cover_pct:.1f}%)")
    
    # Total performance
    total_games = final_df[final_df['total_over'].notna()]
    if not total_games.empty:
        overs = total_games['total_over'].sum()
        total_totals = len(total_games)
        over_pct = (overs / total_totals) * 100
        print(f"   Total record: {overs}-{total_totals-overs} ({over_pct:.1f}%)")
    
    # Show sample data
    print(f"\n🎯 Sample Combined Data:")
    sample = final_df.head(5)
    for _, game in sample.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        week = game['week']
        season = game['season']
        home_pts = int(game['home_points']) if pd.notna(game['home_points']) else 'TBD'
        away_pts = int(game['away_points']) if pd.notna(game['away_points']) else 'TBD'
        
        spread_info = f" ({game['tennessee_spread']})" if pd.notna(game['tennessee_spread']) else ""
        total_info = f" O/U {game['total_points']}" if 'total_points' in game and pd.notna(game['total_points']) else ""
        
        print(f"   {season} Week {week}: {away_team} @ {home_team} ({away_pts}-{home_pts}){spread_info}{total_info}")

if __name__ == "__main__":
    main()
