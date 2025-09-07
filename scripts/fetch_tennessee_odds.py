#!/usr/bin/env python3
"""Fetch betting odds data for Tennessee football games (2022-2024)."""

import requests
import pandas as pd
import sys
sys.path.append('src')
from app.config import settings
from datetime import datetime, timedelta

def main():
    print("🎰 Fetching Betting Odds for Tennessee Games (2022-2024)")
    print("=" * 60)
    
    # Load Tennessee games data
    try:
        games_df = pd.read_csv('tennessee_games_2022_2024.csv')
        print(f"✅ Loaded {len(games_df)} Tennessee games")
    except Exception as e:
        print(f"❌ Error loading games data: {e}")
        return
    
    # Initialize API settings
    odds_api_key = settings.odds_api_key
    headers = {
        'Accept': 'application/json'
    }
    
    print(f"🔑 Using Odds API Key: {odds_api_key[:10]}...")
    
    # Fetch odds for each season
    seasons = [2022, 2023, 2024]
    all_odds = []
    
    for season in seasons:
        print(f"\n📅 Fetching odds for {season} season...")
        
        try:
            # Get games for this season
            season_games = games_df[games_df['season'] == season]
            print(f"   Found {len(season_games)} games for {season}")
            
            # Fetch odds for each game
            for _, game in season_games.iterrows():
                game_id = game['id']
                home_team = game['homeTeam']
                away_team = game['awayTeam']
                week = game['week']
                start_date = pd.to_datetime(game['startDate'])
                
                print(f"   🏈 Week {week}: {away_team} @ {home_team}")
                
                # Fetch odds for this specific game
                odds_data = fetch_game_odds(odds_api_key, headers, game_id, start_date)
                
                if odds_data:
                    all_odds.extend(odds_data)
                    print(f"      ✅ Found {len(odds_data)} odds entries")
                else:
                    print(f"      ⚠️  No odds data found")
                    
        except Exception as e:
            print(f"   ❌ Error fetching odds for {season}: {e}")
    
    # Save results
    if all_odds:
        print(f"\n📊 Combining odds data...")
        odds_df = pd.DataFrame(all_odds)
        
        # Save to CSV
        filename = 'tennessee_odds_2022_2024.csv'
        odds_df.to_csv(filename, index=False)
        
        print(f"✅ Total odds entries found: {len(odds_df)}")
        print(f"💾 Data saved to: {filename}")
        
        # Show summary
        print(f"\n📈 Summary by market:")
        market_summary = odds_df.groupby('market').size()
        for market, count in market_summary.items():
            print(f"   {market}: {count} entries")
        
        # Show sample odds
        print(f"\n🎯 Sample odds data:")
        sample_odds = odds_df.head(5)
        for _, odds in sample_odds.iterrows():
            print(f"   {odds['home_team']} vs {odds['away_team']}: {odds['market']} - {odds['book']}")
            
    else:
        print("❌ No odds data found for Tennessee games")

def fetch_game_odds(api_key, headers, game_id, start_date):
    """Fetch odds for a specific game."""
    odds_data = []
    
    # Define markets to fetch
    markets = ['spreads', 'totals', 'h2h']  # Moneyline, spreads, totals
    
    for market in markets:
        try:
            url = f'https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds'
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': market,
                'dateFormat': 'iso',
                'oddsFormat': 'american',
                'bookmakers': 'fanduel,draftkings,betmgm,caesars'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                odds = response.json()
                
                # Filter for Tennessee games
                for game_odds in odds:
                    home_team = game_odds.get('home_team', '')
                    away_team = game_odds.get('away_team', '')
                    
                    # Check if this is a Tennessee game
                    if 'Tennessee' in home_team or 'Tennessee' in away_team:
                        # Extract odds data
                        for bookmaker in game_odds.get('bookmakers', []):
                            book_name = bookmaker.get('title', '')
                            
                            for market_data in bookmaker.get('markets', []):
                                market_name = market_data.get('key', market)
                                
                                for outcome in market_data.get('outcomes', []):
                                    odds_entry = {
                                        'game_id': game_id,
                                        'home_team': home_team,
                                        'away_team': away_team,
                                        'market': market_name,
                                        'book': book_name,
                                        'outcome': outcome.get('name', ''),
                                        'price': outcome.get('price', 0),
                                        'point': outcome.get('point', None),
                                        'fetched_at': datetime.now().isoformat()
                                    }
                                    odds_data.append(odds_entry)
                                    
            else:
                print(f"      ⚠️  API error for {market}: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error fetching {market}: {e}")
    
    return odds_data

if __name__ == "__main__":
    main()
