#!/usr/bin/env python3
"""
单场预测入口
用法：
  python3 code/predict_match.py "Spain" "France"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.elo_engine import match_probabilities
from models.poisson_model import predict_match
from utils.io import load_teams


def predict(team_a_name: str, team_b_name: str):
    teams = load_teams()
    
    if team_a_name not in teams:
        print(f"❌ 未找到球队: {team_a_name}")
        sys.exit(1)
    if team_b_name not in teams:
        print(f"❌ 未找到球队: {team_b_name}")
        sys.exit(1)
    
    ta = teams[team_a_name]
    tb = teams[team_b_name]
    
    print("=" * 70)
    print(f"⚽ {team_a_name} vs {team_b_name}")
    print("=" * 70)
    print(f"\n基础数据:")
    print(f"  {team_a_name:<14} Elo {ta['elo']}, FIFA #{ta['fifa']}, xG {ta['xg_for']:.2f}/场")
    print(f"  {team_b_name:<14} Elo {tb['elo']}, FIFA #{tb['fifa']}, xG {tb['xg_for']:.2f}/场")
    
    # Elo 模型
    elo_p = match_probabilities(ta["elo"], tb["elo"])
    print(f"\n🔢 Elo 模型预测:")
    print(f"  {team_a_name} 胜: {elo_p['p_win_a']*100:.1f}%")
    print(f"  平局:           {elo_p['p_draw']*100:.1f}%")
    print(f"  {team_b_name} 胜: {elo_p['p_win_b']*100:.1f}%")
    
    # Poisson 模型
    poisson_result = predict_match(ta, tb)
    print(f"\n⚽ Poisson + Dixon-Coles 模型:")
    print(f"  期望进球: {team_a_name} = {poisson_result['lambda_a']}, {team_b_name} = {poisson_result['lambda_b']}")
    print(f"  胜平负: {team_a_name} {poisson_result['outcome']['p_win_a']*100:.1f}% / 平 {poisson_result['outcome']['p_draw']*100:.1f}% / {team_b_name} {poisson_result['outcome']['p_win_b']*100:.1f}%")
    print(f"\n  Top 5 比分:")
    for (a, b), p in poisson_result["top_scorelines"]:
        print(f"    {team_a_name[:6]:>6}-{team_b_name[:6]:<6}  {a}-{b}  ({p*100:.1f}%)")
    
    # 集成（简单平均）
    p_a = (elo_p["p_win_a"] + poisson_result["outcome"]["p_win_a"]) / 2
    p_d = (elo_p["p_draw"] + poisson_result["outcome"]["p_draw"]) / 2
    p_b = (elo_p["p_win_b"] + poisson_result["outcome"]["p_win_b"]) / 2
    
    print(f"\n🎯 集成预测（Elo + Poisson 加权平均）:")
    print(f"  {team_a_name} 胜: {p_a*100:.1f}%")
    print(f"  平局:           {p_d*100:.1f}%")
    print(f"  {team_b_name} 胜: {p_b*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 predict_match.py <球队A> <球队B>")
        print('例: python3 predict_match.py "Spain" "France"')
        sys.exit(1)
    
    predict(sys.argv[1], sys.argv[2])
