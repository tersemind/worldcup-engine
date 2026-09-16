"""
一键自动化数据刷新主脚本

执行流程：
1. 调用 elo_fetcher 联网抓 eloratings.net
2. 提示 Claude 用 WebFetch 抓 Polymarket（聚合到 multi_source_market）
3. 提示 Claude 用 WebSearch 抓伤病情报
4. 输出当前数据快照供验证
5. 运行综合预测

用法：
  python3 auto_refresh.py            # 自动化全刷新（仅 Elo）+ 提示
  python3 auto_refresh.py --status   # 只显示当前数据状态
"""
import sys
import json
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_teams, DATA_RAW
from data.injuries_fetcher import load_injuries, get_health_adjustments


CODE_DIR = Path(__file__).parent.parent  # code/
ENGINE_DIR = Path(__file__).parent.parent.parent  # worldcup-engine/


def show_status():
    """显示当前所有数据源的状态"""
    print("=" * 70)
    print("📊 WorldCup Engine 数据状态总览")
    print("=" * 70)
    
    teams = load_teams()
    
    # 1. Elo / 市场赔率最新更新时间
    file_path = DATA_RAW / "teams.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    last_updated = data.get("_last_updated", {})
    
    elo_dates = [v.get("elo") for v in last_updated.values() if v.get("elo")]
    market_dates = [v.get("market_implied") for v in last_updated.values() if v.get("market_implied")]
    
    print(f"\n📈 Elo 数据：")
    if elo_dates:
        print(f"  最新更新: {max(elo_dates)}")
        print(f"  涉及球队: {len(elo_dates)}")
    else:
        print(f"  ⚠️  尚未抓取过")
    
    print(f"\n💰 市场赔率：")
    if market_dates:
        print(f"  最新更新: {max(market_dates)}")
        print(f"  涉及球队: {len(market_dates)}")
    else:
        print(f"  ⚠️  尚未抓取过")
    
    # 2. 伤病库
    inj_data = load_injuries()
    print(f"\n🏥 伤病库：")
    print(f"  最后更新: {inj_data.get('_updated', 'N/A')}")
    print(f"  涉及球队: {len(inj_data.get('teams', {}))}")
    
    health_adjs = get_health_adjustments()
    if health_adjs:
        print(f"\n  各队 Health 调整：")
        for team, adj in sorted(health_adjs.items(), key=lambda x: x[1]):
            n_inj = len(inj_data["teams"][team].get("injuries", []))
            print(f"    {team:<14} {adj:>+6.2f}pp ({n_inj} 个伤情)")
    
    # 3. 蒙特卡洛输出
    outputs = sorted((ENGINE_DIR / "data" / "outputs").glob("*.json"))
    print(f"\n💾 输出文件：{len(outputs)} 个")
    for f in outputs:
        size_kb = f.stat().st_size / 1024
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
        print(f"  {f.name:<35} {size_kb:>6.1f} KB  ({mtime})")
    
    print("\n" + "=" * 70)


def auto_refresh():
    """自动刷新流程"""
    print("=" * 70)
    print("🚀 WorldCup Engine v1.2 — 一键自动刷新")
    print("=" * 70)
    
    # Step 1: 抓 Elo
    print("\n📡 Step 1/4: 抓取最新 Elo（eloratings.net）")
    print("-" * 70)
    try:
        result = subprocess.run(
            ["python3", str(CODE_DIR / "data" / "elo_fetcher.py")],
            capture_output=True, text=True, timeout=30
        )
        # 输出最后 8 行
        lines = result.stdout.strip().split("\n")
        for line in lines[-8:]:
            print(f"  {line}")
    except Exception as e:
        print(f"  ❌ Elo 抓取失败: {e}")
    
    # Step 2: 提示 Claude 抓市场赔率（不能自动）
    print("\n💰 Step 2/4: 市场赔率刷新")
    print("-" * 70)
    print("  ℹ️  需要 Claude 用 WebFetch 抓取，请运行：")
    print("      /wc-engine refresh market")
    print("  或手动：")
    print("      python3 code/data/refresh_helper.py update <team> market_implied <0.xxx>")
    
    # Step 3: 提示 Claude 抓伤病情报
    print("\n🏥 Step 3/4: 伤病情报刷新")
    print("-" * 70)
    print("  ℹ️  需要 Claude 用 WebSearch 抓取，请运行：")
    print("      /wc-engine refresh injuries")
    print("  当前伤病库摘要：")
    health_adjs = get_health_adjustments()
    if health_adjs:
        for team, adj in sorted(health_adjs.items(), key=lambda x: x[1])[:5]:
            print(f"      {team:<14} {adj:>+6.2f}pp")
    else:
        print("      (空)")
    
    # Step 4: 重跑预测
    print("\n🎲 Step 4/4: 是否立即重跑预测？")
    print("-" * 70)
    print("  运行：")
    print("    python3 code/run_tournament.py 100000      # 基准（27 秒）")
    print("    python3 code/run_scenarios.py 100000        # 三情景（80 秒）")
    
    print("\n" + "=" * 70)
    print("✅ 刷新流程提示完毕")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_status()
    else:
        auto_refresh()
