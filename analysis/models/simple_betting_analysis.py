#!/usr/bin/env python3
"""Simple Betting Analysis to understand the data and find profitable opportunities."""

import pandas as pd
import numpy as np

def main():
    print("💰 Simple Betting Analysis")
    print("=" * 50)
    
    # Load data
    try:
        tennessee_df = pd.read_csv('tennessee_games_2022_2024.csv')
        odds_df = pd.read_csv('tennessee_odds_2022_2024.csv')
        print(f"✅ Loaded {len(tennessee_df)} Tennessee games")
        print(f"✅ Loaded {len(odds_df)} betting lines")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze betting lines data
    print(f"\n📊 Analyzing Betting Lines Data:")
    print("-" * 30)
    
    print(f"Betting lines columns: {list(odds_df.columns)}")
    print(f"Sample betting lines:")
    print(odds_df.head(10))
    
    print(f"\nMarkets available: {odds_df['market'].unique()}")
    print(f"Books available: {odds_df['book'].unique()}")
    
    # Filter for spread bets
    spread_odds = odds_df[odds_df['market'] == 'spreads'].copy()
    print(f"\nSpread bets: {len(spread_odds)}")
    
    # Group by game
    game_groups = spread_odds.groupby('game_id')
    print(f"Games with spread data: {len(game_groups)}")
    
    # Analyze each game
    print(f"\n🎯 Game-by-Game Analysis:")
    print("-" * 30)
    
    profitable_opportunities = []
    
    for game_id, game_odds in game_groups:
        # Find the corresponding Tennessee game
        tennessee_game = tennessee_df[tennessee_df['id'] == int(game_id)]
        
        if len(tennessee_game) == 0:
            continue
            
        game = tennessee_game.iloc[0]
        opponent = game['awayTeam'] if game['homeTeam'] == 'Tennessee' else game['homeTeam']
        
        # Get Tennessee line
        tennessee_lines = game_odds[game_odds['outcome'].str.contains('Tennessee', na=False)]
        
        if len(tennessee_lines) == 0:
            continue
            
        # Get the best line for Tennessee
        best_line = tennessee_lines.loc[tennessee_lines['point'].abs().idxmax()]
        tennessee_line = best_line['point']
        tennessee_price = best_line['price']
        tennessee_book = best_line['book']
        
        # Calculate actual result
        tennessee_home = game['homeTeam'] == 'Tennessee'
        tennessee_points = game['homePoints'] if tennessee_home else game['awayPoints']
        opponent_points = game['awayPoints'] if tennessee_home else game['homePoints']
        actual_diff = tennessee_points - opponent_points
        tennessee_won = tennessee_points > opponent_points
        
        # Calculate if Tennessee covered
        tennessee_covered = actual_diff > tennessee_line
        
        # Simple prediction based on ELO difference
        elo_diff = game['homePregameElo'] - game['awayPregameElo']
        if not tennessee_home:
            elo_diff = -elo_diff  # Flip if Tennessee is away
        
        # Calculate edge
        ml_pred_diff = elo_diff / 25  # Rough conversion from ELO to points
        edge = ml_pred_diff - tennessee_line
        
        # Determine bet recommendation
        bet_recommendation = "NO BET"
        bet_amount = 0
        actual_profit = 0
        
        if abs(edge) > 3:  # Minimum edge threshold
            if edge > 0:  # ML predicts Tennessee covers
                bet_recommendation = "BET TENNESSEE"
                bet_amount = 100
                if tennessee_covered:
                    actual_profit = bet_amount * 0.91  # -110 odds
                else:
                    actual_profit = -bet_amount
            else:  # ML predicts Tennessee doesn't cover
                bet_recommendation = "BET OPPONENT"
                bet_amount = 100
                if not tennessee_covered:
                    actual_profit = bet_amount * 0.91  # -110 odds
                else:
                    actual_profit = -bet_amount
        
        profitable_opportunities.append({
            'game_id': game_id,
            'opponent': opponent,
            'week': game['week'],
            'season': game['season'],
            'tennessee_line': tennessee_line,
            'ml_pred_diff': ml_pred_diff,
            'edge': edge,
            'bet_recommendation': bet_recommendation,
            'bet_amount': bet_amount,
            'actual_diff': actual_diff,
            'tennessee_covered': tennessee_covered,
            'actual_profit': actual_profit,
            'book': tennessee_book
        })
        
        # Print analysis
        home_away = "vs" if tennessee_home else "@"
        print(f"   Week {game['week']} ({game['season']}): {home_away} {opponent}")
        print(f"      Line: {tennessee_line:+.1f} ({tennessee_book})")
        print(f"      ML Pred: {ml_pred_diff:+.1f}")
        print(f"      Edge: {edge:+.1f} points")
        print(f"      Recommendation: {bet_recommendation}")
        if bet_amount > 0:
            print(f"      Bet Amount: ${bet_amount:.0f}")
            print(f"      Actual Result: {actual_diff:+.1f} points")
            print(f"      Covered: {'Yes' if tennessee_covered else 'No'}")
            print(f"      Actual Profit: ${actual_profit:+.2f}")
        print()
    
    # Calculate overall performance
    print(f"\n📊 Overall Betting Performance:")
    print("-" * 30)
    
    total_bets = len([b for b in profitable_opportunities if b['bet_amount'] > 0])
    total_profit = sum(b['actual_profit'] for b in profitable_opportunities)
    winning_bets = len([b for b in profitable_opportunities if b['actual_profit'] > 0])
    
    if total_bets > 0:
        win_rate = winning_bets / total_bets
        roi = (total_profit / (total_bets * 100)) * 100
        
        print(f"   Total Bets: {total_bets}")
        print(f"   Total Profit: ${total_profit:+.2f}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   ROI: {roi:+.2f}%")
        
        # Analyze by season
        for season in [2022, 2023, 2024]:
            season_bets = [b for b in profitable_opportunities if b['season'] == season and b['bet_amount'] > 0]
            if season_bets:
                season_profit = sum(b['actual_profit'] for b in season_bets)
                season_win_rate = len([b for b in season_bets if b['actual_profit'] > 0]) / len(season_bets)
                print(f"   {season} Season: {len(season_bets)} bets | Profit: ${season_profit:+.2f} | Win Rate: {season_win_rate:.1%}")
        
        # Analyze by bet type
        tennessee_bets = [b for b in profitable_opportunities if b['bet_recommendation'] == 'BET TENNESSEE']
        opponent_bets = [b for b in profitable_opportunities if b['bet_recommendation'] == 'BET OPPONENT']
        
        if tennessee_bets:
            tennessee_profit = sum(b['actual_profit'] for b in tennessee_bets)
            tennessee_win_rate = len([b for b in tennessee_bets if b['actual_profit'] > 0]) / len(tennessee_bets)
            print(f"   Tennessee Bets: {len(tennessee_bets)} | Profit: ${tennessee_profit:+.2f} | Win Rate: {tennessee_win_rate:.1%}")
        
        if opponent_bets:
            opponent_profit = sum(b['actual_profit'] for b in opponent_bets)
            opponent_win_rate = len([b for b in opponent_bets if b['actual_profit'] > 0]) / len(opponent_bets)
            print(f"   Opponent Bets: {len(opponent_bets)} | Profit: ${opponent_profit:+.2f} | Win Rate: {opponent_win_rate:.1%}")
    
    # Save results
    if profitable_opportunities:
        bets_df = pd.DataFrame(profitable_opportunities)
        bets_df.to_csv('simple_betting_analysis.csv', index=False)
        print(f"\n💾 Results saved to: simple_betting_analysis.csv")
    
    # Betting recommendations
    print(f"\n🎯 Betting Recommendations:")
    print("-" * 30)
    
    if total_profit > 0:
        print(f"✅ PROFITABLE STRATEGY IDENTIFIED!")
        print(f"   Focus on bets with edge > 3 points")
        print(f"   Target games with clear ELO advantages")
        print(f"   Consider home field advantage")
    else:
        print(f"❌ Strategy needs improvement")
        print(f"   Consider higher edge thresholds")
        print(f"   Focus on most confident predictions")
        print(f"   Avoid betting on close games")

if __name__ == "__main__":
    main()
