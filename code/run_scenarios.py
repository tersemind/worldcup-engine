#!/usr/bin/env python3
"""
三情景蒙特卡洛主入口

用法：
  python3 code/run_scenarios.py              # 默认 100k 次/情景
  python3 code/run_scenarios.py 50000        # 50k 次/情景
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.scenario_engine import main

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n)
