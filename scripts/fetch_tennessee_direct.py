#!/usr/bin/env python3
"""Fetch Tennessee football games from 2022 to present using direct API calls."""

import requests
import pandas as pd
import sys
sys.path.append('src')
from app.config import settings

def main():
    print("🏈 Fetching Tennessee Football Games (2022-Present)")
    print("=" * 50)
    
    api_key = settings.cfbd_api_key
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    print(f"🔑 Using API Key: {api_key[:10]}...")
    
    # Fetch games for each season
    seasons = [2022, 2023, 2024]
    all_games = []
    
    for season in seasons:
        print(f"\n📅 Fetching {season} season games...")
        
        try:
            # Search for Tennessee games
            url = 'https://api.collegefootballdata.com/games'
            params = {'year': season, 'team': 'Tennessee'}
            response = requests.get(url, headers=headers, params=params)
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                games = response.json()
                print(f"   ✅ Found {len(games)} Tennessee games for {season}")
                
                if games:
                    # Convert to DataFrame
                    df = pd.DataFrame(games)
                    df['season'] = season
                    all_games.append(df)
                    
                    # Show sample games
                    print("   Sample games:")
                    for i, game in enumerate(games[:3]):
                        home_team = game.get('home_team', 'Unknown')
                        away_team = game.get('away_team', 'Unknown')
                        week = game.get('week', '?')
                        home_points = game.get('home_points')
                        away_points = game.get('away_points')
                        
                        if home_points is not None and away_points is not None:
                            score = f"({away_points}-{home_points})"
                        else:
                            score = "(TBD)"
                            
                        print(f"     Week {week}: {away_team} @ {home_team} {score}")
                else:
                    print(f"   ⚠️  No games found for {season}")
            else:
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
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
        recent_games = combined_games.tail(5)
        for _, game in recent_games.iterrows():
            home_team = game.get('home_team', 'Unknown')
            away_team = game.get('away_team', 'Unknown')
            week = game.get('week', '?')
            home_points = game.get('home_points')
            away_points = game.get('away_points')
            
            if home_points is not None and away_points is not None:
                score = f"({away_points}-{home_points})"
            else:
                score = "(TBD)"
                
            print(f"   {game['season']} Week {week}: {away_team} @ {home_team} {score}")
        
        # Show columns available
        print(f"\n📋 Available data columns:")
        print(f"   {list(combined_games.columns)}")
            
    else:
        print("❌ No games found for Tennessee in the specified seasons")

if __name__ == "__main__":
    main()
