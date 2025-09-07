#!/usr/bin/env python3
"""Fetch Tennessee football games from 2022 to present."""

import sys
import os
sys.path.append('src')

from app.providers.cfbd_client import CFBDClient
import pandas as pd
from datetime import datetime

def main():
    print("🏈 Fetching Tennessee Football Games (2022-Present)")
    print("=" * 50)
    
    # Initialize CFBD client
    try:
        client = CFBDClient()
        print(f"✅ CFBD Client initialized")
        print(f"🔑 API Key: {client.api_key[:10]}...")
    except Exception as e:
        print(f"❌ Error initializing CFBD client: {e}")
        return
    
    # Test connection with teams
    try:
        print("\n🔍 Testing API connection...")
        teams = client.fetch_teams()
        print(f"✅ Successfully fetched {len(teams)} teams")
        
        # Look for Tennessee
        tennessee_teams = teams[teams['team'].str.contains('Tennessee', case=False, na=False)]
        if not tennessee_teams.empty:
            print("\n📋 Tennessee teams found:")
            for _, team in tennessee_teams.iterrows():
                print(f"   - {team['team']} ({team['conference']})")
        else:
            print("⚠️  No Tennessee teams found in teams list")
            
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return
    
    # Fetch games for each season
    seasons = [2022, 2023, 2024]
    all_games = []
    
    print(f"\n📅 Fetching games for seasons: {seasons}")
    
    for season in seasons:
        try:
            print(f"\n🏈 Fetching {season} season games...")
            games = client.fetch_games(season=season, team='Tennessee')
            
            if not games.empty:
                all_games.append(games)
                print(f"✅ Found {len(games)} games for {season}")
                
                # Show sample games
                print("   Sample games:")
                sample_games = games[['week', 'home', 'away', 'home_points', 'away_points', 'completed']].head(3)
                for _, game in sample_games.iterrows():
                    home_team = game['home']
                    away_team = game['away']
                    home_pts = game['home_points'] if pd.notna(game['home_points']) else 'TBD'
                    away_pts = game['away_points'] if pd.notna(game['away_points']) else 'TBD'
                    status = "✅" if game['completed'] else "⏳"
                    print(f"   {status} Week {game['week']}: {away_team} @ {home_team} ({away_pts}-{home_pts})")
            else:
                print(f"⚠️  No games found for {season}")
                
        except Exception as e:
            print(f"❌ Error fetching games for {season}: {e}")
    
    # Combine and save results
    if all_games:
        print(f"\n📊 Combining results...")
        combined_games = pd.concat(all_games, ignore_index=True)
        
        # Sort by season and week
        combined_games = combined_games.sort_values(['season', 'week'])
        
        # Save to CSV
        filename = 'tennessee_games_2022_2024.csv'
        combined_games.to_csv(filename, index=False)
        
        print(f"✅ Total Tennessee games found: {len(combined_games)}")
        print(f"💾 Data saved to: {filename}")
        
        # Show summary
        print(f"\n📈 Summary by season:")
        season_summary = combined_games.groupby('season').size()
        for season, count in season_summary.items():
            print(f"   {season}: {count} games")
        
        # Show recent games
        print(f"\n🏆 Recent games:")
        recent_games = combined_games.tail(5)[['season', 'week', 'home', 'away', 'home_points', 'away_points']]
        for _, game in recent_games.iterrows():
            home_team = game['home']
            away_team = game['away']
            home_pts = game['home_points'] if pd.notna(game['home_points']) else 'TBD'
            away_pts = game['away_points'] if pd.notna(game['away_points']) else 'TBD'
            print(f"   {game['season']} Week {game['week']}: {away_team} @ {home_team} ({away_pts}-{home_pts})")
            
    else:
        print("❌ No games found for Tennessee in the specified seasons")

if __name__ == "__main__":
    main()
