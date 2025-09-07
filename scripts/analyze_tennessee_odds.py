#!/usr/bin/env python3
"""Analyze Tennessee betting odds data."""

import pandas as pd
import numpy as np

def main():
    print("🎰 Tennessee Betting Odds Analysis (2022-2024)")
    print("=" * 60)
    
    # Load odds data
    try:
        odds_df = pd.read_csv('tennessee_odds_2022_2024.csv')
        print(f"✅ Loaded {len(odds_df)} odds entries")
    except Exception as e:
        print(f"❌ Error loading odds data: {e}")
        return
    
    # Load games data for context
    try:
        games_df = pd.read_csv('tennessee_games_2022_2024.csv')
        print(f"✅ Loaded {len(games_df)} games for context")
    except Exception as e:
        print(f"❌ Error loading games data: {e}")
        return
    
    # Basic summary
    print(f"\n📊 Odds Data Summary:")
    print(f"   Total entries: {len(odds_df)}")
    print(f"   Unique games: {odds_df['game_id'].nunique()}")
    print(f"   Bookmakers: {odds_df['book'].nunique()}")
    print(f"   Markets: {odds_df['market'].nunique()}")
    
    # Market breakdown
    print(f"\n📈 Market Breakdown:")
    market_counts = odds_df['market'].value_counts()
    for market, count in market_counts.items():
        print(f"   {market}: {count} entries")
    
    # Bookmaker breakdown
    print(f"\n🏪 Bookmaker Breakdown:")
    book_counts = odds_df['book'].value_counts()
    for book, count in book_counts.head(5).items():
        print(f"   {book}: {count} entries")
    
    # Sample odds for recent games
    print(f"\n🎯 Sample Odds (Recent Games):")
    recent_games = odds_df.tail(20)
    for _, odds in recent_games.iterrows():
        home_team = odds['home_team']
        away_team = odds['away_team']
        market = odds['market']
        book = odds['book']
        outcome = odds['outcome']
        price = odds['price']
        point = odds['point']
        
        if pd.notna(point):
            print(f"   {away_team} @ {home_team}: {market} - {outcome} {point} ({price}) [{book}]")
        else:
            print(f"   {away_team} @ {home_team}: {market} - {outcome} ({price}) [{book}]")
    
    # Analyze spreads for Tennessee games
    print(f"\n📏 Tennessee Spread Analysis:")
    tennessee_spreads = odds_df[
        (odds_df['market'] == 'spreads') & 
        (odds_df['outcome'].str.contains('Tennessee', case=False, na=False))
    ]
    
    if not tennessee_spreads.empty:
        print(f"   Found {len(tennessee_spreads)} Tennessee spread bets")
        
        # Show recent spreads
        recent_spreads = tennessee_spreads.tail(10)
        for _, spread in recent_spreads.iterrows():
            home_team = spread['home_team']
            away_team = spread['away_team']
            point = spread['point']
            price = spread['price']
            book = spread['book']
            
            if 'Tennessee' in home_team:
                print(f"   Tennessee vs {away_team}: {point} ({price}) [{book}]")
            else:
                print(f"   {home_team} vs Tennessee: {point} ({price}) [{book}]")
    
    # Analyze totals
    print(f"\n📊 Totals Analysis:")
    totals = odds_df[odds_df['market'] == 'totals']
    if not totals.empty:
        print(f"   Found {len(totals)} total bets")
        
        # Show recent totals
        recent_totals = totals.tail(10)
        for _, total in recent_totals.iterrows():
            home_team = total['home_team']
            away_team = total['away_team']
            point = total['point']
            price = total['price']
            book = total['book']
            outcome = total['outcome']
            
            print(f"   {away_team} @ {home_team}: Total {outcome} {point} ({price}) [{book}]")
    
    # Moneyline analysis
    print(f"\n💰 Moneyline Analysis:")
    moneylines = odds_df[odds_df['market'] == 'h2h']
    if not moneylines.empty:
        print(f"   Found {len(moneylines)} moneyline bets")
        
        # Show recent moneylines
        recent_ml = moneylines.tail(10)
        for _, ml in recent_ml.iterrows():
            home_team = ml['home_team']
            away_team = ml['away_team']
            price = ml['price']
            book = ml['book']
            outcome = ml['outcome']
            
            print(f"   {away_team} @ {home_team}: {outcome} ({price}) [{book}]")
    
    print(f"\n✅ Analysis complete! Data saved to tennessee_odds_2022_2024.csv")

if __name__ == "__main__":
    main()
