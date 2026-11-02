"""数据加载与预处理：8760 小时负荷/光伏/分时电价、基准售价、无项目成本。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from src import config as cfg

_MONTHS = {
    "一月": 1,
    "二月": 2,
    "三月": 3,
    "四月": 4,
    "五月": 5,
    "六月": 6,
    "七月": 7,
    "八月": 8,
    "九月": 9,
    "十月": 10,
    "十一月": 11,
    "十二月": 12,
}
_CACHE: dict | None = None


def load_data(refresh: bool = False, *, tou_multiplier: float = 1.0) -> dict:
    """加载规划数据；可按同一倍率调整全部逐时电量电价。

    倍率只作用于分时电量电价，不改变最大需量电价。绿电结算、
    电网购电、无项目电费与基准光伏时段电价使用同一调整后序列。
    原始缓存保持不变，避免相邻敏感性情景互相污染。
    """
    if not np.isfinite(tou_multiplier) or tou_multiplier <= 0.0:
        raise ValueError("tou_multiplier 必须为正的有限数")
    global _CACHE
    if _CACHE is None or refresh:
        load = _load()
        pv = _pv()
        tou, month = _tou()
        base = _base_sell(pv, tou)
        energy = float(np.dot(load, tou))
        demand = 0.0
        for m in range(1, 13):
            peak = float(load[month == m].max())
            demand += cfg.DEMAND_PEAK * peak * cfg.DEMAND_CHARGE
        _CACHE = {
            "load": load,
            "pv": pv,
            "tou": tou,
            "month": month,
            "base_sell": base,
            "no_project": energy + demand,
        }
    if tou_multiplier == 1.0:
        return _CACHE
    raw = _CACHE
    scaled_tou = raw["tou"] * tou_multiplier
    demand_cost = raw["no_project"] - float(np.dot(raw["load"], raw["tou"]))
    return {
        **raw,
        "tou": scaled_tou,
        "base_sell": _base_sell(raw["pv"], scaled_tou),
        "no_project": float(np.dot(raw["load"], scaled_tou)) + demand_cost,
    }


def _load() -> np.ndarray:
    f = pd.read_excel(cfg.RAW_LOAD_FILE, sheet_name=0, header=0, usecols=[0, 1])
    f.columns = ["h", "kw"]
    h = pd.to_numeric(f["h"], errors="coerce").to_numpy()
    kw = pd.to_numeric(f["kw"], errors="coerce").to_numpy(float)
    if not np.array_equal(h, np.arange(1, cfg.N + 1)):
        raise ValueError("负荷小时序号必须连续覆盖1到8760")
    if not np.isfinite(kw).all() or (kw < 0).any():
        raise ValueError("负荷功率含非法值")
    return kw


def _pv() -> np.ndarray:
    f = pd.read_excel(cfg.PV_PROFILE_FILE, sheet_name=0, header=0, usecols=[0, 1])
    if len(f) != cfg.N:
        raise ValueError("光伏小时功率不是8760条")
    p = pd.to_numeric(f.iloc[:, 1], errors="coerce").to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or p.max() <= 0:
        raise ValueError("光伏小时功率含非法值")
    return p / _pv_ref_capacity()


def _pv_ref_capacity() -> float:
    f = pd.read_excel(cfg.PV_REF_FILE, sheet_name=0, header=0)
    names = f.iloc[:, 0].astype(str).str.strip().to_numpy()
    hits = np.flatnonzero(names == cfg.PV_REF_NAME)
    if len(hits) != 1:
        raise ValueError(f"装机容量表中应唯一找到：{cfg.PV_REF_NAME}")
    cap = float(pd.to_numeric(f.iloc[hits[0], 4], errors="coerce"))
    if cap <= 0:
        raise ValueError("参考光伏装机容量必须为正数")
    return cap


def _tou() -> tuple[np.ndarray, np.ndarray]:
    f = pd.read_excel(cfg.TOU_FILE, sheet_name=0, header=None)
    rows = f.iloc[1:13, :25]
    months: list[int] = []
    for raw in rows.iloc[:, 0].astype(str).str.strip():
        num = pd.to_numeric(raw, errors="coerce")
        if pd.notna(num):
            months.append(int(num))
        elif raw in _MONTHS:
            months.append(_MONTHS[raw])
        elif raw.endswith("月") and raw[:-1].isdigit():
            months.append(int(raw[:-1]))
        else:
            raise ValueError(f"无法识别分时电价月份标签：{raw!r}")
    if months != list(range(1, 13)):
        raise ValueError("分时电价月份列必须按1到12顺序排列")
    mtx = rows.iloc[:, 1:25].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if mtx.shape != (12, 24) or not np.isfinite(mtx).all():
        raise ValueError("分时电价表必须包含12×24个有效价格")
    month_id = np.repeat(np.arange(1, 13), np.asarray(cfg.MONTH_DAYS) * 24)
    hourly = np.concatenate(
        [np.tile(mtx[m - 1], cfg.MONTH_DAYS[m - 1]) for m in range(1, 13)]
    )
    return hourly, month_id


def _base_sell(pv: np.ndarray, tou: np.ndarray) -> float:
    """365个日度光伏时段 TOU 算术平均，四舍五入4位。

    gamma 是相对 TOU 的绝对价格系数，基准 8 折由
    cfg.BASELINE_GAMMA 表示，不在这里重复乘入。
    """
    daily = []
    for d in range(365):
        s, e = d * 24, (d + 1) * 24
        act = np.flatnonzero(pv[s:e] > 0)
        if act.size == 0:
            raise ValueError(f"第{d + 1}天没有正光伏出力")
        daily.append(float(np.mean(tou[s:e][act[0] : act[-1] + 1])))
    return round(float(np.mean(daily)), 4)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化的结果类型：{type(value).__name__}")


def write_json(path: Path, value) -> None:
    """原子保存计算结果；支持逐时数组，不把绘图逻辑引入计算模块。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("rows 不能为空")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


@lru_cache(maxsize=64)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    """文件未变化时复用摘要，避免每个 LHS 状态都重复读取原始数据。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> str:
    """将缓存绑定到源码、输入数据和当前参数，路径迁移不改变数据身份。"""
    digest = hashlib.sha256()
    paths = [
        cfg.RAW_LOAD_FILE, cfg.PV_PROFILE_FILE, cfg.PV_REF_FILE, cfg.TOU_FILE,
        *(Path(__file__).parent / name for name in
          ("config.py", "data.py", "model.py", "experiment1.py", "experiment2.py")),
    ]
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns).encode())
    parameters = {
        key: value for key, value in vars(cfg).items()
        if key.isupper() and isinstance(value, (int, float, str, bool, tuple, type(None)))
    }
    digest.update(json.dumps(parameters, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
