"""
通用数据 IO 工具模块
"""
import json
import os
import warnings
from pathlib import Path

# 抑制 LibreSSL 警告
warnings.filterwarnings("ignore", message=".*LibreSSL.*")
warnings.filterwarnings("ignore", category=Warning)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS = ROOT / "data" / "outputs"
LOGS = ROOT / "logs"


def load_teams():
    """加载 48 支球队基础数据"""
    with open(DATA_RAW / "teams.json", "r") as f:
        return json.load(f)["teams"]


def load_groups():
    """加载 12 个小组分组"""
    with open(DATA_RAW / "groups.json", "r") as f:
        return json.load(f)["groups"]


def load_groups_full():
    """加载完整分组数据（含 pathway）"""
    with open(DATA_RAW / "groups.json", "r") as f:
        return json.load(f)


def save_output(name: str, data: dict):
    """保存输出到 outputs 目录"""
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = DATA_OUTPUTS / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def get_qualified_teams():
    """返回参赛 48 队列表（排除未晋级球队）"""
    teams = load_teams()
    return {name: data for name, data in teams.items() if data.get("group") != "_"}
