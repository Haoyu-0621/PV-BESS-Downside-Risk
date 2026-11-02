"""核心模型：8760 小时光储 MILP。

- 弹性求解 solve_planning：给定 (α,γ) 重新优化容量，15 年 NPV 目标
  （单代表年 + 第2~15年退化折减组装，口径与《实验方案设计.md》一致）；
- 规划求解启用工程可行集半连续状态：PV∈{0}∪[200,3000]、ESS∈{0}∪[100,2000]、
  默认禁 ESS-only（y_ESS≤y_PV）、固定工程费已删除（投资 = 容量线性成本）；
  锁定求解容量已给定，不施加进入门槛（门槛决定是否建设，不决定能否运行）；
- 结算口径（EMC 统一绿电折扣）：用户承担负荷购电，投资方承担电网给电池
  充电的购电成本；光伏直供和储能放电按 γ×tou 向用户结算。电网充电费
  直接从投资方项目现金流中扣除，投资方实际 NPV 与规划 NPV一致；
  需量节省按 DEMAND_SHARE 由用户支付给投资方（削峰收益分享）；余电上网收益归投资方；
- 消纳率 ≥ 70% 为规划期"上网代理约束"（仅规划求解强制；锁定评估可选），
  上网功率上限默认读取 EXPORT_MAX_KW，也可由锁定压力状态单独传入；
- 构建向量化（MVar 成批加约束），保留充放电互斥二进制（MILP，MUTEX=True，
  严格禁止同小时同时充放）；
- audit 内嵌物理与经济独立重构检查。
"""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.data import load_data

_EC = np.sqrt(cfg.ROUNDTRIP)
_ED = np.sqrt(cfg.ROUNDTRIP)
_N = cfg.N


def _capacity_spec() -> tuple[float, float, float, float, bool, float]:
    """返回容量边界与储能半连续进入约束参数。

    储能进入功率为正时，储能仍可不安装；若安装，则名义功率
    ``C_RATE * E_ESS`` 不得低于进入阈值。
    """
    bounds = (
        cfg.PV_MIN,
        cfg.PV_MAX,
        cfg.ESS_MIN,
        cfg.ESS_MAX,
        cfg.ESS_ENTRY_POWER_KW,
        cfg.C_RATE,
    )
    if not all(np.isfinite(v) for v in bounds):
        raise ValueError("容量边界、储能进入功率与 C_RATE 必须为有限数")
    if not 0.0 <= cfg.PV_MIN <= cfg.PV_MAX:
        raise ValueError("光伏容量边界必须满足 0 <= PV_MIN <= PV_MAX")
    if not 0.0 <= cfg.ESS_MIN <= cfg.ESS_MAX:
        raise ValueError("储能容量边界必须满足 0 <= ESS_MIN <= ESS_MAX")
    if cfg.C_RATE <= 0.0:
        raise ValueError("C_RATE 必须为正数")
    if not 0.0 <= cfg.ESS_ENTRY_POWER_KW <= cfg.C_RATE * cfg.ESS_MAX:
        raise ValueError("储能进入功率必须位于 [0, C_RATE * ESS_MAX]")
    needs_binary = cfg.ESS_ENTRY_POWER_KW > 0.0
    ess_lb = 0.0 if needs_binary else cfg.ESS_MIN
    min_installed_kwh = (
        max(cfg.ESS_MIN, cfg.ESS_ENTRY_POWER_KW / cfg.C_RATE) if needs_binary else 0.0
    )
    return cfg.PV_MIN, cfg.PV_MAX, ess_lb, cfg.ESS_MAX, needs_binary, min_installed_kwh


def _entry_spec() -> tuple[bool, float, bool, float, bool]:
    """工程可行集参数（进入/筛选求解专用，见《实验方案设计.md》1.3）。

    返回 (needs_pv_binary, pv_min_entry_kw, needs_ess_binary, min_ess_kwh, allow_ess_only)。
    半连续投资状态：光伏 ∈ {0} ∪ [PV_MIN_ENTRY_KW, PV_MAX]；
    储能 ∈ {0} ∪ [min_ess_kwh, ESS_MAX]（min_ess_kwh 取能量门槛与
    ESS_ENTRY_POWER_KW/C_RATE 的较大者，0.5C 派生功率不构成独立下界）。
    默认禁止 ESS-only：y_ESS <= y_PV（合同要求光储一体时另设 ALLOW_ESS_ONLY 语义）。
    仅在规划类求解（capacity=None）中启用；锁定求解容量已给定，
    进入门槛决定"是否建设"、不决定"能否运行"。
    """
    if not 0.0 <= cfg.PV_MIN_ENTRY_KW <= cfg.PV_MAX:
        raise ValueError("光伏进入门槛必须满足 0 <= PV_MIN_ENTRY_KW <= PV_MAX")
    if not 0.0 <= cfg.ESS_MIN_ENTRY_KWH <= cfg.ESS_MAX:
        raise ValueError("储能进入门槛必须满足 0 <= ESS_MIN_ENTRY_KWH <= ESS_MAX")
    needs_ess_binary = cfg.ESS_MIN_ENTRY_KWH > 0.0 or cfg.ESS_ENTRY_POWER_KW > 0.0
    needs_pv_binary = cfg.PV_MIN_ENTRY_KW > 0.0
    if needs_ess_binary and not cfg.ALLOW_ESS_ONLY and not needs_pv_binary:
        raise ValueError("禁止 ESS-only 时必须有正的光伏进入门槛（y_ESS <= y_PV 需要 y_PV 存在）")
    min_ess_kwh = (
        max(cfg.ESS_MIN_ENTRY_KWH, cfg.ESS_ENTRY_POWER_KW / cfg.C_RATE)
        if needs_ess_binary
        else 0.0
    )
    return needs_pv_binary, cfg.PV_MIN_ENTRY_KW, needs_ess_binary, min_ess_kwh, cfg.ALLOW_ESS_ONLY


def npv_at(cashflows: np.ndarray, rate: float) -> float:
    return float(np.sum(cashflows / (1.0 + rate) ** np.arange(cashflows.size)))


def irr(cashflows: np.ndarray, tol: float = 1e-10) -> float | None:
    """直接 IRR：二分法；现金流结构异常返回 None。"""
    if cashflows[0] >= 0 or not (cashflows[1:] > 0).any():
        return None
    lo, hi = -1.0 + 1e-9, 1.0
    v_lo, v_hi = npv_at(cashflows, lo), npv_at(cashflows, hi)
    while np.signbit(v_lo) == np.signbit(v_hi) and hi < 1e6:
        hi = 2.0 * hi + 1.0
        v_hi = npv_at(cashflows, hi)
    if np.signbit(v_lo) == np.signbit(v_hi):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        v_mid = npv_at(cashflows, mid)
        if abs(v_mid) <= tol or abs(hi - lo) <= tol:
            return float(mid)
        if np.signbit(v_lo) == np.signbit(v_mid):
            lo, v_lo = mid, v_mid
        else:
            hi, v_hi = mid, v_mid
    return float(0.5 * (lo + hi))


def err(cashflows: np.ndarray) -> float | None:
    """ERR：正现金流按再投资利率终值化、负现金流按融资利率现值化。"""
    periods = cashflows.size - 1
    fv = sum(
        max(cf, 0.0) * (1 + cfg.R_REQ) ** (periods - p)
        for p, cf in enumerate(cashflows)
    )
    pv = sum(-min(cf, 0.0) / (1 + cfg.R_REQ) ** p for p, cf in enumerate(cashflows))
    if fv <= 0 or pv <= 0:
        return None
    return float((fv / pv) ** (1 / periods) - 1)


def _resolve_pricing(
    pricing_rule: str | None,
    arbitrage_share: float | None,
) -> tuple[str, float]:
    rule = cfg.PRICING_RULE if pricing_rule is None else str(pricing_rule)
    if rule != cfg.PRICING_RULE_TOU_SERVICE:
        raise ValueError(f"invalid pricing_rule: {rule!r}")
    share = cfg.ARBITRAGE_SHARE if arbitrage_share is None else float(arbitrage_share)
    if not np.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("arbitrage_share must be in [0, 1]")
    return rule, share


def _pricing_components(
    *,
    pricing_rule: str | None,
    arbitrage_share: float | None,
    gamma: float,
    base_sell: float,
    tou: np.ndarray,
    pv_to_user,
    ess_to_user,
) -> dict:
    """投资方结算（用户电表 + EMC统一绿电折扣口径，2026-09 定稿）。

    - 光伏直供 pv_to_user：按使用时刻 gamma × tou 结算；
    - 储能放电按放电时刻 gamma × tou 统一结算；
    - 电网给电池充电的电费由投资方支付，并直接从投资方现金流中扣除；
    - 本版本不设置套利分成（arbitrage_saving/fee 恒 0）。

    输入可以是 NumPy 数组或 Gurobi 线性表达式。
    """
    rule, share = _resolve_pricing(pricing_rule, arbitrage_share)
    # gamma 是相对目录 TOU 的绝对价格/折扣系数；0.8 表示 8 折，
    # 由调用方在基准点传入 gamma=cfg.BASELINE_GAMMA。
    unit = gamma
    pv_settlement = unit * (tou * pv_to_user).sum()
    # 当前 EMC 简化口径：储能放电统一作为储能绿电服务结算量。
    ess_settlement = unit * (tou * ess_to_user).sum()
    return {
        "pricing_rule": rule,
        "arbitrage_share": share,
        "pv_settlement": pv_settlement,
        "ess_settlement": ess_settlement,
        "arbitrage_saving": 0.0,
        "arbitrage_fee": 0.0,
        "total_settlement": pv_settlement + ess_settlement,
    }


def _solve(
    *,
    alpha: float,
    gamma: float,
    pricing_rule: str | None,
    arbitrage_share: float | None,
    discount_rate: float | None,
    capacity: tuple[float, float] | None,
    degrade: float,
    tou_multiplier: float = 1.0,
    annual_only: bool = False,
    include_user: bool = True,
    require_build: bool = False,
    investment_state: str = "free",
    enforce_self_use: bool = True,
    export_max_kw: float | None = None,
    threads: int | None = None,
    allow_uncertified: bool = False,
    fixed_pv: float | None = None,
    fixed_ess: float | None = None,
    time_limit: float | None = None,
) -> dict:
    pricing_rule, arbitrage_share = _resolve_pricing(pricing_rule, arbitrage_share)
    rate = cfg.R_REQ if discount_rate is None else discount_rate
    solver_threads = cfg.THREADS if threads is None else threads
    if not isinstance(solver_threads, int) or isinstance(solver_threads, bool) or solver_threads < 1:
        raise ValueError("threads 必须为正整数")
    solve_time_limit = cfg.TIME_LIMIT if time_limit is None else float(time_limit)
    if not np.isfinite(solve_time_limit) or solve_time_limit <= 0.0:
        raise ValueError("time_limit 必须为正数")
    if not np.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError("discount_rate 必须位于 [0,1)")
    pv_degrade = float(getattr(cfg, "PV_DEGRADE", 0.0))
    if not np.isfinite(pv_degrade) or not 0.0 <= pv_degrade < 1.0:
        raise ValueError("PV_DEGRADE 必须位于 [0,1)")
    if require_build and capacity is not None:
        raise ValueError("require_build 仅用于规划求解（capacity=None）")
    if investment_state not in {"free", "pv_only", "pv_battery"}:
        raise ValueError("invalid investment_state")
    if capacity is not None and investment_state != "free":
        raise ValueError("investment_state requires planning mode")
    if require_build and investment_state != "free":
        raise ValueError("require_build and investment_state are mutually exclusive")
    if (fixed_pv is None) == (fixed_ess is None) and (fixed_pv is not None or fixed_ess is not None):
        raise ValueError("半固定容量需且仅需固定一个维度（fixed_pv 或 fixed_ess 之一）")
    if capacity is not None and (fixed_pv is not None or fixed_ess is not None):
        raise ValueError("capacity 与半固定参数互斥")
    half_fixed = fixed_pv is not None or fixed_ess is not None
    if half_fixed and investment_state != "free":
        raise ValueError("半固定容量仅支持 investment_state='free'")
    export_limit = cfg.EXPORT_MAX_KW if export_max_kw is None else float(export_max_kw)
    if export_limit is not None and (
        not np.isfinite(export_limit) or export_limit <= 0.0
    ):
        raise ValueError("export_max_kw 必须为正数或 None")

    pv_min, pv_max, ess_lb, ess_max, _, _ = _capacity_spec()
    data = load_data(tou_multiplier=tou_multiplier)
    load, pv, tou = data["load"], data["pv"], data["tou"]
    month = data["month"]
    base_sell, no_project = data["base_sell"], data["no_project"]
    s_load = alpha * load

    import gurobipy as gp

    m = gp.Model("lean")
    m.Params.Threads = solver_threads
    m.Params.OutputFlag = cfg.VERBOSE
    m.Params.TimeLimit = solve_time_limit
    m.Params.MIPGap = cfg.MIP_GAP
    m.Params.MIPGapAbs = cfg.MIP_GAP_ABS
    # 对偶单纯形（E1 阶段验证过的提速）：LP 12s vs 默认 60s；
    # 兼有 MUTEX 二进制时为 MILP，不影响正确性，只影响求解速度。
    method = getattr(cfg, "TOU_METHOD", cfg.METHOD) if pricing_rule == cfg.PRICING_RULE_TOU_SERVICE else cfg.METHOD
    if method != 0:
        m.Params.Method = method
    if pricing_rule == cfg.PRICING_RULE_TOU_SERVICE:
        # 聚焦寻找可行 incumbent，配合零投资 start 缩短“先找到正 NPV、再证明”的时间。
        m.Params.MIPFocus = getattr(cfg, "TOU_MIP_FOCUS", 1)

    f = m.addMVar((_N, 7), lb=0.0)
    pv_u, pv_e, g_e, e_u, g_u, ex, cu = (
        f[:, 0],
        f[:, 1],
        f[:, 2],
        f[:, 3],
        f[:, 4],
        f[:, 5],
        f[:, 6],
    )
    soc = m.addMVar((_N,), lb=0.0, name="soc")
    storage_forced_off = investment_state == "pv_only" or (
        capacity is not None and capacity[1] <= 0.0
    )
    z = None
    mode_capacity = None
    d = m.addMVar((12,), lb=0.0)
    cpv = m.addMVar((1,), lb=pv_min, ub=pv_max)
    eess = m.addMVar((1,), lb=ess_lb, ub=ess_max)
    needs_pv_binary = needs_ess_binary = False  # 规划分支内由 _entry_spec 覆盖
    y_pv = y_ess = None
    if capacity is not None:
        m.addConstr(cpv == capacity[0])
        m.addConstr(eess == capacity[1])
        # 锁定求解：容量已给定，不施加进入门槛（门槛决定"是否建设"，不决定"能否运行"）；
        # 固定工程费已删除：不投资候选 (0,0) 的 NPV = 0，V_s(0) = 0。
        build = 1.0 if (capacity[0] > 0.0 or capacity[1] > 0.0) else 0.0
        y_pv_expr = float(build)
    elif half_fixed:
        # 半固定容量（条件联合优化）：固定一个维度，另一维度走工程可行集
        # （PV ∈ {0}∪[PV_MIN_ENTRY_KW, PV_MAX]，ESS ∈ {0}∪[ESS_MIN_ENTRY_KWH, ESS_MAX]）。
        needs_pv_binary, pv_min_entry_kw, needs_ess_binary, min_ess_kwh, allow_ess_only = (
            _entry_spec()
        )
        if fixed_pv is not None:
            # 固定光伏：ESS 半连续自由优化（0 或 ≥ min_ess_kwh）
            m.addConstr(cpv == fixed_pv)
            y_pv_expr = 1.0 if fixed_pv > 0.0 else 0.0
            if needs_ess_binary:
                y_ess = m.addVar(vtype=gp.GRB.BINARY, name="ess_installed")
                m.addConstr(eess >= min_ess_kwh * y_ess, name="ess_entry_min")
                m.addConstr(eess <= ess_max * y_ess, name="ess_entry_off")
                if not allow_ess_only:
                    m.addConstr(y_ess <= y_pv_expr, name="ess_requires_pv")
        else:
            # 固定储能：PV 半连续自由优化（0 或 ≥ pv_min_entry_kw）
            m.addConstr(eess == fixed_ess)
            if needs_pv_binary:
                y_pv = m.addVar(vtype=gp.GRB.BINARY, name="pv_installed")
                m.addConstr(cpv <= pv_max * y_pv, name="pv_entry_off")
                m.addConstr(cpv >= pv_min_entry_kw * y_pv, name="pv_entry_min")
                y_pv_expr = y_pv
            else:
                y_pv_expr = 1.0
            if needs_ess_binary and not allow_ess_only and needs_pv_binary:
                # 固定 ESS>0 时 y_ess 由 eess 上界隐含；PV 二进制作用于 cpv，
                # y_ess <= y_pv 由固定 ESS 的 E 与 PV 门槛交互，这里不再显式加
                # （固定储能允许光伏为 0 时无意义，由结果 NPV 决策）。
                pass
    else:
        # 规划求解：工程可行集（半连续投资状态）
        needs_pv_binary, pv_min_entry_kw, needs_ess_binary, min_ess_kwh, allow_ess_only = (
            _entry_spec()
        )
        if needs_pv_binary:
            y_pv = m.addVar(vtype=gp.GRB.BINARY, name="pv_installed")
            m.addConstr(cpv <= pv_max * y_pv, name="pv_entry_off")
            m.addConstr(cpv >= pv_min_entry_kw * y_pv, name="pv_entry_min")
            y_pv_expr = y_pv
        else:
            y_pv_expr = 1.0
        if needs_ess_binary:
            y_ess = m.addVar(vtype=gp.GRB.BINARY, name="ess_installed")
            m.addConstr(eess >= min_ess_kwh * y_ess, name="ess_entry_min")
            m.addConstr(eess <= ess_max * y_ess, name="ess_entry_off")
            if not allow_ess_only:
                m.addConstr(y_ess <= y_pv_expr, name="ess_requires_pv")
        state_requires_pv = require_build or investment_state in {"pv_only", "pv_battery"}
        state_requires_ess = investment_state == "pv_battery"
        state_forbids_ess = investment_state == "pv_only"
        if state_requires_pv and needs_pv_binary:
            m.addConstr(y_pv == 1, name="build_required")
        elif state_requires_pv:
            m.addConstr(cpv >= max(pv_min, pv_min_entry_kw), name="build_required")
        if needs_ess_binary:
            if state_requires_ess:
                m.addConstr(y_ess == 1, name="battery_required")
            elif state_forbids_ess:
                m.addConstr(y_ess == 0, name="battery_forbidden")
        elif state_requires_ess:
            m.addConstr(eess >= min_ess_kwh, name="battery_required")
        elif state_forbids_ess:
            m.addConstr(eess == 0.0, name="battery_forbidden")

    m.addConstr(g_u + pv_u + e_u == s_load)
    m.addConstr(pv_u + pv_e + ex + cu == cpv * pv)
    # 自发自用、余电上网：当小时可用的 PV 先直供负荷；只有覆盖负荷后，
    # 剩余 PV 才可充电、上网或弃光。否则低合同价可能诱使模型一边购电供负荷，
    # 一边把原本可直供的 PV 以更高的上网电价卖出。
    # 已知 PV 容量时直接固定逐时直供量，避免给大批锁定情景增加整数变量。
    known_pv_kw = capacity[0] if capacity is not None else fixed_pv
    if known_pv_kw is not None:
        m.addConstr(pv_u == np.minimum(s_load, float(known_pv_kw) * pv), name="pv_direct_first")
    else:
        pv_surplus = m.addMVar((_N,), lb=0.0, name="pv_surplus_after_load")
        load_remainder = m.addMVar((_N,), lb=0.0, name="load_after_pv")
        m.addConstr(pv_surplus == pv_e + ex + cu, name="pv_surplus_balance")
        m.addConstr(load_remainder == g_u + e_u, name="load_remainder_balance")
        for i in range(_N):
            m.addSOS(
                gp.GRB.SOS_TYPE1,
                [pv_surplus[i].item(), load_remainder[i].item()],
                [1.0, 2.0],
            )
    nxt = np.append(np.arange(1, _N), 0)
    m.addConstr(soc[nxt] == soc + _EC * (pv_e + g_e) - e_u / _ED)
    # 放电只能使用小时开始时已有的电量；严格互斥时这是物理方程的强化，
    # 连续对照时也可避免“同小时先充后放”的虚假能量。
    m.addConstr(e_u <= _ED * soc, name="discharge_from_pre_soc")
    ue, up = degrade * eess, cfg.C_RATE * degrade * eess
    m.addConstr(soc >= cfg.SOC_LO * ue)
    m.addConstr(soc <= cfg.SOC_HI * ue)
    # pv_e、g_e、e_u 非负，且其和受 up 限制，因此三个单独上界是冗余约束。
    # 删除它们不改变可行域，但可减少约 26,280 条约束，尤其有利于新规则 MILP。
    m.addConstr(pv_e + g_e + e_u <= up)
    # ---- 同时充放惩罚（LP 化后的软约束替代 MUTEX）----
    # 无二进制时模型可以"同小时既充又放"（LP 松弛漂移）。物理上不可能，
    # 且会虚增套利结算。用线性重叠量做小惩罚：
    #   overlap_t ≥ (充功率_t + 放功率_t) − up_t，overlap_t ≥ 0
    # 由于共享变流器约束 pv_e+g_e+e_u ≤ up，充放之和超出 up 的部分恰好
    # 是"同小时同时充放"的量。惩罚系数 PEN_SIMULT 显著小于真实峰谷价差
    # （0.1~0.5 元/kWh），只消除无物理意义的同小时充放，不干扰正常套利。
    simult_pen = 0.0
    if cfg.MUTEX is False and not storage_forced_off:
        pen = cfg.PEN_SIMULT
        if pen > 0.0:
            overlap = m.addMVar((_N,), lb=0.0, name="simult_overlap")
            # up 是 1 元 MVar（degrade * C_RATE * eess），取 up[0] 作为统一上界
            up_scalar = up[0]
            for i in range(_N):
                m.addConstr(
                    overlap[i] >= (pv_e[i] + g_e[i] + e_u[i]) - up_scalar,
                    name="simult_overlap_def",
                )
            simult_pen = pen * overlap.sum()
    if cfg.MUTEX and not storage_forced_off:
        mutex_formulation = getattr(cfg, "MUTEX_FORMULATION", "binary")
        if mutex_formulation == "indicator":
            # 原生 indicator 与共享变流器上限共同构成严格互斥：
            # z=1 为充电模式时放电为 0；z=0 为放电模式时充电为 0。
            # 不再需要 mode_capacity 及其 McCormick 包络，减少连续变量和线性行。
            z = m.addMVar((_N,), vtype=gp.GRB.BINARY, name="storage_mode")
            if y_ess is not None:
                m.addConstr(z <= y_ess, name="mode_requires_ess")
            for i in range(_N):
                m.addGenConstrIndicator(
                    z[i], 1, e_u[i] == 0.0, name=f"discharge_off_in_charge_{i}"
                )
                m.addGenConstrIndicator(
                    z[i], 0, pv_e[i] + g_e[i] == 0.0,
                    name=f"charge_off_in_discharge_{i}",
                )
            soc_bound = cfg.SOC_HI * degrade * ess_max
            soc_change = soc[nxt] - soc
            m.addConstr(
                soc_change >= -soc_bound * (1.0 - z),
                name="soc_charge_mode_monotone",
            )
            m.addConstr(
                soc_change <= soc_bound * z,
                name="soc_discharge_mode_monotone",
            )
        elif mutex_formulation == "sos":
            # 两组 SOS1 只禁止充电与放电同小时发生，不禁止 PV 与电网
            # 同时充电；因此与原 binary formulation 的可行域一致。
            for i in range(_N):
                m.addSOS(gp.GRB.SOS_TYPE1, [pv_e[i], e_u[i]], [1.0, 2.0])
                m.addSOS(gp.GRB.SOS_TYPE1, [g_e[i], e_u[i]], [1.0, 2.0])
        else:
            z = m.addMVar((_N,), vtype=gp.GRB.BINARY)
            if y_ess is not None:
                # ESS 未安装时 z 没有物理意义；联动可消除 LP 中的
                # 无效模式自由度，z∈{0,1} 下不改变可行域。
                m.addConstr(z <= y_ess, name="mode_requires_ess")
            # 强化原 big-M：mode_capacity = eess × z 的 McCormick 包络。
            # z∈{0,1} 时与 charge/discharge 互斥约束完全等价；LP 松弛时
            # 同时保留容量—模式联动，避免固定 M 造成过松的根节点上界。
            mode_capacity = m.addMVar((_N,), lb=0.0, ub=ess_max, name="mode_capacity")
            m.addConstr(mode_capacity <= eess, name="mode_capacity_cap")
            m.addConstr(mode_capacity <= ess_max * z, name="mode_capacity_on")
            m.addConstr(
                mode_capacity >= eess - ess_max * (1 - z),
                name="mode_capacity_off",
            )
            power_scale = cfg.C_RATE * degrade
            m.addConstr(pv_e + g_e <= power_scale * mode_capacity, name="charge_mode")
            m.addConstr(
                e_u <= power_scale * (eess - mode_capacity),
                name="discharge_mode",
            )
            # 模式—SOC 单调性强化：z=1 为充电模式时 SOC 不下降，z=0
            # 为放电模式时 SOC 不上升。由原 SOC 方程和互斥约束严格推出，
            # 但可明显收紧 z fractional 时的根节点松弛。
            soc_bound = cfg.SOC_HI * degrade * ess_max
            soc_change = soc[nxt] - soc
            m.addConstr(
                soc_change >= -soc_bound * (1.0 - z),
                name="soc_charge_mode_monotone",
            )
            m.addConstr(
                soc_change <= soc_bound * z,
                name="soc_discharge_mode_monotone",
            )

    # 消纳率约束 = 规划期的"上网代理约束"（限制光伏无限制上网，代理未知的上网限制）。
    # 锁定评估（capacity 已定）取消强制：容量无法改变，约束失去配比意义，
    # 消纳率改作报告指标（<70% 即"低消纳/合同指标违反区"）。
    if enforce_self_use:
        m.addConstr((pv_u + pv_e).sum() >= cfg.MIN_SELF_USE * (cpv * pv).sum())

    # 上网功率上限（并网容量，物理约束）：规划与锁定均适用；None = 不限制。
    # 默认值由 F1 校准（名义设计最大小时上网功率）填入，敏感性 {0.5×,1×,1.5×}。
    if export_limit is not None:
        m.addConstr(ex <= export_limit, name="export_cap")

    base_d = np.asarray(
        [cfg.DEMAND_PEAK * s_load[month == mo].max() for mo in range(1, 13)]
    )
    for mo in range(1, 13):
        idx = np.flatnonzero(month == mo)
        m.addConstr(d[mo - 1] <= base_d[mo - 1])
        m.addConstr(
            d[mo - 1] * np.ones(idx.size) >= cfg.DEMAND_PEAK * (g_u[idx] + g_e[idx])
        )

    # ---- 单年经济（第 1 年 / 单年度口径）----
    # 固定工程费：2026-08 定稿删除（投资 = 容量线性成本）；FIXED_ENG=0 保持该口径，
    # >0 时作为敏感性/对照（一次性入场费会压低最优容量）。
    inv = cfg.FIXED_ENG + cfg.PV_CAPEX * cpv + cfg.ESS_CAPEX * eess
    sell = cfg.sell_price(gamma, base_sell)
    green1 = (pv_u + pv_e).sum()  # 光伏自用量：直供负荷 + 光伏充电
    pricing1 = _pricing_components(
        pricing_rule=pricing_rule,
        arbitrage_share=arbitrage_share,
        gamma=gamma,
        base_sell=base_sell,
        tou=tou,
        pv_to_user=pv_u,
        ess_to_user=e_u,
    )
    er1 = pricing1["total_settlement"]
    xr1 = cfg.EXPORT_PRICE * ex.sum()
    # 用户承担负荷购电 g_u；电网给储能充电 g_e 由投资方承担。
    # 因此 g_e 不计入用户账单，但作为投资方真实现金支出扣除。
    gc1 = (tou * g_u).sum()
    # 电网充电费是投资方真实现金支出，直接从项目现金流中扣除。
    chg1 = (tou * g_e).sum()
    dc1 = cfg.DEMAND_CHARGE * d.sum()
    sv1 = float(base_d.sum()) - d.sum()
    dr1 = cfg.DEMAND_SHARE * cfg.DEMAND_CHARGE * sv1
    om = cfg.PV_OM * cpv + cfg.ESS_OM_RATE * cfg.ESS_CAPEX * eess
    cf1 = er1 + xr1 + dr1 - chg1 - om - simult_pen
    us1 = er1 + gc1 + dc1 + dr1
    charging_costs = [chg1]

    dy_vars = []
    if annual_only:
        m.setObjective(cf1, gp.GRB.MAXIMIZE)
    else:
        # ---- 第 2~15 年：退化折减组装（与单代表年调度同源）----
        cfs, uss = [cf1], [us1]
        charging_costs = [chg1]
        for y in range(2, cfg.LIFE + 1):
            k = (1.0 - cfg.DEGRADE) ** (y - 1)
            pv_k = (1.0 - pv_degrade) ** (y - 1)
            # PV充入电池同时受组件衰减和储能可用功率衰减约束。
            # 两者中较小者进入电池，剩余的PV转为上网，保持年度能量守恒。
            pv_charge_k = min(pv_k, k)
            pv_charge_excess_k = pv_k - pv_charge_k
            dy = m.addMVar((12,), lb=0.0)
            dy_vars.append(dy)
            for mo in range(1, 13):
                idx = np.flatnonzero(month == mo)
                m.addConstr(dy[mo - 1] <= base_d[mo - 1])
                m.addConstr(
                    dy[mo - 1] * np.ones(idx.size)
                    >= cfg.DEMAND_PEAK
                    * (
                        g_u[idx]
                        + (1 - pv_k) * pv_u[idx]
                        + k * g_e[idx]
                        + (1 - k) * e_u[idx]
                    )
                )
            pricing = _pricing_components(
                pricing_rule=pricing_rule,
                arbitrage_share=arbitrage_share,
                gamma=gamma,
                base_sell=base_sell,
                tou=tou,
                pv_to_user=pv_k * pv_u,
                ess_to_user=k * e_u,
            )
            er = pricing["total_settlement"]
            xr = cfg.EXPORT_PRICE * (
                pv_k * ex + pv_charge_excess_k * pv_e
            ).sum()
            # 用户承担负荷购电、光伏衰减缺口和储能退化缺口；退化后的
            # 电网充电 k×g_e 由投资方承担，不计入用户账单。
            gc = (
                tou
                * (g_u + (1 - pv_k) * pv_u + (1 - k) * e_u)
            ).sum()
            chg = (tou * (k * g_e)).sum()
            dc = cfg.DEMAND_CHARGE * dy.sum()
            dr = cfg.DEMAND_SHARE * cfg.DEMAND_CHARGE * (float(base_d.sum()) - dy.sum())
            # 同时充放惩罚按年退化系数缩放（第 y 年充放流 = k × 代表年流）。
            # 电网充电费属于投资方现金支出，已从项目现金流中扣除。
            cfs.append(er + xr + dr - chg - om - k * simult_pen)
            charging_costs.append(chg)
            uss.append(er + gc + dc + dr)
        if include_user:
            m.addConstr(sum(uss) <= cfg.LIFE * alpha * no_project)
        salvage = cfg.SALVAGE * inv
        obj = -inv + salvage / (1 + rate) ** cfg.LIFE
        for y, cf in enumerate(cfs, start=1):
            obj = obj + cf / (1 + rate) ** y
        m.setObjective(obj, gp.GRB.MAXIMIZE)

    # ---- 零投资 MIP 起始解 ----
    # 自由规划包含显然可行的 (PV,ESS)=(0,0) 候选，其 NPV=0。
    # 默认启发式有时会先找到负 NPV 的建设 incumbent，严重拖慢分支界定。
    # 给出完整可行 start 只改善搜索顺序，不改变任何约束、目标或最终认证标准。
    if (
        capacity is None
        and not half_fixed
        and investment_state == "free"
        and not require_build
    ):
        f_start = np.zeros((_N, 7), dtype=float)
        f_start[:, 4] = s_load  # g_u
        f.Start = f_start
        soc.Start = np.zeros(_N, dtype=float)
        d.Start = base_d
        cpv.Start = np.zeros(1, dtype=float)
        eess.Start = np.zeros(1, dtype=float)
        if y_pv is not None:
            y_pv.Start = 0.0
        if y_ess is not None:
            y_ess.Start = 0.0
        if z is not None:
            z.Start = np.zeros(_N, dtype=float)
        if mode_capacity is not None:
            mode_capacity.Start = np.zeros(_N, dtype=float)
        for dy in dy_vars:
            dy.Start = base_d

    # ---- 连续松弛取整 warm start（仅新规则自由规划） ----
    # 先解一次 LP，按 LP 模式取整，再固定整数变量解连续可行子问题，
    # 将所得完整可行解作为正式 MILP 的 MIP start。最终仍恢复原整数
    # 类型并由原模型认证；该过程只改变搜索起点，不改变可行域/目标。
    warm_start_enabled = (
        pricing_rule == cfg.PRICING_RULE_TOU_SERVICE
        and getattr(cfg, "MUTEX_FORMULATION", "binary") == "binary"
        and getattr(cfg, "TOU_LP_WARM_START", False)
        and capacity is None
        and not half_fixed
        and investment_state == "free"
        and not require_build
    )
    if warm_start_enabled:
        m.update()
        all_vars = list(m.getVars())
        integer_vars = [
            v for v in all_vars if v.VType in (gp.GRB.BINARY, gp.GRB.INTEGER)
        ]
        original_int = [(v, v.VType, v.LB, v.UB) for v in integer_vars]
        warm_values = None
        try:
            # ---- LP relaxation ----
            for v, _, _, _ in original_int:
                v.VType = gp.GRB.CONTINUOUS
            m.Params.TimeLimit = min(solve_time_limit, 60.0)
            m.update()
            m.optimize()
            if m.SolCount > 0:
                rounded = [1.0 if v.X >= 0.5 else 0.0 for v, _, _, _ in original_int]
                # ---- 固定取整后的模式/安装状态，求连续可行 dispatch ----
                for (v, _, _, _), value in zip(original_int, rounded):
                    v.LB = value
                    # 直接使用 value；LB 的更新是延迟提交的，读取 v.LB
                    # 可能仍返回旧值，从而产生 lower > upper 的坏边界。
                    v.UB = value
                m.update()
                m.optimize()
                if m.SolCount > 0:
                    warm_values = [float(v.X) for v in all_vars]
        except (gp.GurobiError, ValueError, IndexError):
            warm_values = None
        finally:
            for v, vtype, lb, ub in original_int:
                v.VType = vtype
                v.LB = lb
                v.UB = ub
            m.Params.TimeLimit = solve_time_limit
            m.update()
        if warm_values is not None:
            for v, value in zip(all_vars, warm_values):
                v.Start = value

    m.optimize()
    time_limited = m.Status == gp.GRB.TIME_LIMIT

    # 边界扫描需要经过认证的最优值。限时得到的 incumbent 只能用于诊断，
    # 不能被标记为 optimal，否则其 NPV 正负号可能制造虚假边界。
    certified = m.Status == gp.GRB.OPTIMAL
    solve_status = "optimal"
    if not certified:
        if m.Status == gp.GRB.INFEASIBLE:
            status = "infeasible"
        elif m.SolCount > 0:
            status = "solver_not_certified"
        else:
            status = "no_solution"
        if not allow_uncertified or status != "solver_not_certified":
            return {
                "status": status,
                "solver_status": int(m.Status),
                "time_limited": bool(time_limited),
                "sol_count": int(m.SolCount),
            }
        solve_status = status

    def val(e):
        return float(e.getValue()) if hasattr(e, "getValue") else float(e)

    cf1_v = val(cf1)
    us1_v = val(us1)
    pricing_v = {
        key: val(pricing1[key])
        for key in ("pv_settlement", "ess_settlement", "arbitrage_saving", "arbitrage_fee")
    }

    if annual_only:
        npv_v = cf1_v
        npv_actual_v = cf1_v
        charging_cost_pv = val(chg1)
        full_actual = None
        full = None
        npv_recon = 0.0
        us_total_v = us1_v
        outside = float(alpha * no_project)
        annual_user_costs_v = (float(us1_v),)
        annual_outside_costs_v = (float(outside),)
    else:
        cf_v = np.asarray([val(c) for c in cfs])
        charging_cost_v = np.asarray([val(c) for c in charging_costs])
        inv_v = val(inv)
        full = np.concatenate(([-inv_v], cf_v))
        full[-1] += val(cfg.SALVAGE * inv)
        full_actual = full
        npv_v = float(m.ObjVal)
        npv_actual_v = npv_v
        charging_cost_pv = float(
            np.sum(charging_cost_v / (1.0 + rate) ** np.arange(1, cfg.LIFE + 1))
        )
        npv_recon = abs(npv_v - npv_at(full, rate))
        annual_user_costs_v = tuple(float(val(u)) for u in uss)
        annual_outside_costs_v = tuple(
            float(alpha * no_project) for _ in range(cfg.LIFE)
        )
        us_total_v = float(sum(annual_user_costs_v))
        outside = float(cfg.LIFE * alpha * no_project)

    gen = float(cpv[0].X * pv.sum())
    if capacity is None:
        pv_installed = (
            bool(y_pv_expr.X > 0.5)
            if needs_pv_binary and hasattr(y_pv_expr, "X")
            else (float(y_pv_expr) > 0.5)
            if isinstance(y_pv_expr, (int, float))
            else (float(cpv[0].X) > 0.0)
        )
        ess_installed = (
            bool(y_ess.X > 0.5)
            if needs_ess_binary and y_ess is not None
            else (float(eess[0].X) > 0.0)
        )
    else:
        pv_installed = ess_installed = None
    dispatch = {
        "pv_to_user": pv_u.X,
        "pv_to_ess": pv_e.X,
        "grid_to_ess": g_e.X,
        "ess_to_user": e_u.X,
        "grid_to_user": g_u.X,
        "export": ex.X,
        "curtail": cu.X,
        "soc": soc.X,
    }
    aud = audit(
        dispatch,
        s_load,
        pv_gen=float(cpv[0].X) * pv,
        soc_lo=cfg.SOC_LO * val(ue),
        soc_hi=cfg.SOC_HI * val(ue),
        up=val(up),
        base_d=base_d,
        month=month,
        sell=sell,
        tou=tou,
        base_sell=base_sell,
        gamma=gamma,
        pricing_rule=pricing_rule,
        arbitrage_share=arbitrage_share,
        pv_kw=float(cpv[0].X),
        ess_kwh=float(eess[0].X),
        user_model=us1_v,
        cf1_model=cf1_v,
        charging_cost_model=val(chg1),
        entry_mode=capacity is None,
        pv_installed=pv_installed,
        ess_installed=ess_installed,
        invest=val(inv),
    )

    # ---- 后处理判定：容量配置过小 → 投资方不愿进入 → 强制判不投（不可投区）----
    # 仅适用于"自由规划求解"（capacity=None 且 investment_state="free"、非 require_build）：
    # 优化出的光伏/储能容量若低于最小建设门槛（<200 kW / <100 kWh），
    # 意味着"适配负荷的容量太小、投资方不愿意进入"——按不投资处理（容量=0、NPV=0）。
    # 锁定评估（capacity 给定）与强制状态边界求解（pv_only/pv_battery/require_build）
    # 不触发此判定：前者容量已定，后者是"边界探针"，需要真实的强制状态 NPV。
    if (
        capacity is None
        and investment_state == "free"
        and not require_build
        and pv_installed
        and (float(cpv[0].X) < cfg.PV_MIN_ENTRY_KW - 1e-6)
    ):
        pv_kw = 0.0
        ess_kwh = 0.0
        pv_installed = False
        ess_installed = False
        npv_v = 0.0
        full = None
        # 不投资：无投资、无年度现金流（NPV=0 与不投资候选一致）；用户成本 = 无项目成本
        inv_v = 0.0
        result = {
            "status": solve_status,
            "solver_status": int(m.Status),
            "time_limited": bool(time_limited),
            "sol_count": int(m.SolCount),
            "pv_kw": pv_kw,
            "ess_kwh": ess_kwh,
            "npv": npv_v,
            "npv_planning": npv_v,
            "npv_actual_investor": npv_v,
            "grid_charging_cost_pv": 0.0,
            "annual_only": annual_only,
            "irr": None,
            "err": None,
            "invest": inv_v,
            "pv_installed": pv_installed,
            "ess_installed": ess_installed,
            "investment_state": "not_invest",
            "cashflows": (),
            "annual_cashflow": 0.0,
            "annual_user_cost": float(outside),
            "pricing_rule": pricing_rule,
            "arbitrage_share": arbitrage_share,
            "pv_settlement_revenue": 0.0,
            "ess_settlement_revenue": 0.0,
            "arbitrage_saving": 0.0,
            "arbitrage_service_fee": 0.0,
            "user_cost_project": float(outside),
            "user_cost_outside": float(outside),
            "user_saving": 0.0,
            "green_energy": 0.0,
            "pv_generation": 0.0,
            "absorption": 0.0,
            "export_cap_kw": export_limit,
            "export_peak_kw": 0.0,
            "export_cap_binding": False,
            "feasible": True,
            "npv_reconciliation_error": 0.0,
            "dispatch": dispatch,
            "audit": aud,
            "entry_too_small": True,
        }
        return result

    result = {
        "status": solve_status,
        "solver_status": int(m.Status),
        "time_limited": bool(time_limited),
        "sol_count": int(m.SolCount),
        "pv_kw": float(cpv[0].X),
        "ess_kwh": float(eess[0].X),
        "npv": npv_v,
        # npv 为投资方实际项目现金流口径，已扣除电池电网充电成本。
        "npv_planning": npv_v,
        "npv_actual_investor": npv_actual_v,
        "grid_charging_cost_pv": charging_cost_pv,
        "annual_only": annual_only,
        "irr": irr(full) if full is not None else None,
        "irr_actual_investor": irr(full_actual) if full_actual is not None else None,
        "err": err(full) if full is not None else None,
        "invest": val(inv),
        "pv_installed": pv_installed,
        "ess_installed": ess_installed,
        "investment_state": (
            "pv_battery"
            if pv_installed and ess_installed
            else "pv_only"
            if pv_installed
            else "not_invest"
            if capacity is None
            else "locked"
        ),
        "cashflows": tuple(float(v) for v in full) if full is not None else (),
        "cashflows_actual_investor": (
            tuple(float(v) for v in full_actual) if full_actual is not None else ()
        ),
        "annual_cashflow": cf1_v,
        "annual_user_cost": us1_v,
        "annual_user_costs": annual_user_costs_v,
        "annual_outside_costs": annual_outside_costs_v,
        "pricing_rule": pricing_rule,
        "arbitrage_share": arbitrage_share,
        "pv_settlement_revenue": pricing_v["pv_settlement"],
        "ess_settlement_revenue": pricing_v["ess_settlement"],
        "arbitrage_saving": pricing_v["arbitrage_saving"],
        "arbitrage_service_fee": pricing_v["arbitrage_fee"],
        "user_cost_project": us_total_v,
        "user_cost_outside": outside,
        "user_saving": outside - us_total_v,
        "green_energy": val(green1),
        "pv_generation": gen,
        "absorption": float(val(green1) / max(gen, 1e-12)),
        "export_cap_kw": export_limit,
        "export_peak_kw": float(ex.X.max()),
        "export_cap_binding": bool(
            export_limit is not None
            and float(ex.X.max()) >= export_limit - 1e-6
        ),
        "feasible": npv_v >= -1e-6,
        "npv_reconciliation_error": npv_recon,
        "dispatch": dispatch,
        "audit": aud,
    }
    return result


def audit(
    dispatch: dict,
    s_load: np.ndarray,
    *,
    pv_gen: np.ndarray,
    soc_lo: float,
    soc_hi: float,
    up: float,
    base_d: np.ndarray,
    month: np.ndarray,
    sell: float,
    tou: np.ndarray,
    pv_kw: float,
    ess_kwh: float,
    user_model: float,
    cf1_model: float,
    charging_cost_model: float = 0.0,
    base_sell: float | None = None,
    gamma: float = 1.0,
    pricing_rule: str | None = None,
    arbitrage_share: float | None = None,
    entry_mode: bool = False,
    pv_installed: bool | None = None,
    ess_installed: bool | None = None,
    invest: float | None = None,
) -> dict:
    """物理与经济独立重构检查（替代旧诊断框架）。"""
    a = {k: np.asarray(v, float) for k, v in dispatch.items()}
    g_u, g_e, e_u = a["grid_to_user"], a["grid_to_ess"], a["ess_to_user"]
    pv_u, pv_e, ex, cu, soc = (
        a["pv_to_user"],
        a["pv_to_ess"],
        a["export"],
        a["curtail"],
        a["soc"],
    )
    load_res = g_u + pv_u + e_u - s_load
    pv_res = pv_u + pv_e + ex + cu - pv_gen
    direct_priority_res = pv_u - np.minimum(s_load, pv_gen)
    soc_next = np.roll(soc, -1)
    soc_res = soc_next - (soc + _EC * (pv_e + g_e) - e_u / _ED)
    soc_wrap = soc[0] - (soc[-1] + _EC * (pv_e[-1] + g_e[-1]) - e_u[-1] / _ED)
    simult = int((((pv_e + g_e) > 1e-3) & (e_u > 1e-3)).sum())
    charging_cost_re = float((tou * g_e).sum())

    d_implied = np.asarray(
        [
            cfg.DEMAND_PEAK * np.max(g_u[month == m] + g_e[month == m])
            for m in range(1, 13)
        ]
    )
    green = float((pv_u + pv_e).sum())
    price_base = sell if base_sell is None else base_sell
    price_gamma = 1.0 if base_sell is None else gamma
    pricing = _pricing_components(
        pricing_rule=pricing_rule,
        arbitrage_share=arbitrage_share,
        gamma=price_gamma,
        base_sell=price_base,
        tou=tou,
        pv_to_user=pv_u,
        ess_to_user=e_u,
    )
    user_re = (
        float(pricing["total_settlement"])
        + float((tou * g_u).sum())
        + cfg.DEMAND_CHARGE * float(d_implied.sum())
        + cfg.DEMAND_SHARE
        * cfg.DEMAND_CHARGE
        * (float(base_d.sum()) - float(d_implied.sum()))
    )
    cf1_re = (
        float(pricing["total_settlement"])
        + cfg.EXPORT_PRICE * float(ex.sum())
        + cfg.DEMAND_SHARE
        * cfg.DEMAND_CHARGE
        * (float(base_d.sum()) - float(d_implied.sum()))
        - charging_cost_re
        - (cfg.PV_OM * pv_kw + cfg.ESS_OM_RATE * cfg.ESS_CAPEX * ess_kwh)
    )
    self_use = green / max(float(pv_gen.sum()), 1e-12)

    if entry_mode:
        pv_inst, ess_inst = bool(pv_installed), bool(ess_installed)
        entry_consistent = (
            (pv_kw == 0.0) == (not pv_inst)
            and (ess_kwh == 0.0) == (not ess_inst)
            and (not ess_inst or pv_inst)
        )
    else:
        entry_consistent = None
    if invest is not None:
        built = (
            (pv_installed or ess_installed)
            if entry_mode
            else (pv_kw > 0.0 or ess_kwh > 0.0)
        )
        expected_inv = cfg.PV_CAPEX * pv_kw + cfg.ESS_CAPEX * ess_kwh
        fixed_cost_ok = abs(invest - expected_inv) <= 1e-3
    else:
        fixed_cost_ok = None

    return {
        "max_load_balance_error_kw": float(np.max(np.abs(load_res))),
        "max_pv_balance_error_kw": float(np.max(np.abs(pv_res))),
        "max_pv_direct_priority_error_kw": float(np.max(np.abs(direct_priority_res))),
        "simultaneous_export_grid_to_user_steps": int(
            (((ex > 1e-3) & (g_u > 1e-3))).sum()
        ),
        "max_soc_transition_error_kwh": float(np.max(np.abs(soc_res))),
        "soc_min_violation": float(max(0.0, -np.min(soc - soc_lo))),
        "soc_max_violation": float(max(0.0, np.max(soc - soc_hi))),
        "soc_wrap_error_kwh": float(abs(soc_wrap)),
        # 主模型不做来源追踪；保留旧审计键以兼容已有输出读取逻辑。
        "max_source_discharge_decomposition_error_kw": None,
        "max_pv_source_soc_transition_error_kwh": None,
        "max_grid_source_soc_transition_error_kwh": None,
        "max_source_soc_sum_error_kwh": None,
        "pv_source_soc_wrap_error_kwh": None,
        "grid_source_soc_wrap_error_kwh": None,
        "pv_origin_discharge_kwh": None,
        "grid_origin_discharge_kwh": None,
        "simultaneous_chg_dis_steps": int(simult),
        "grid_charging_energy_kwh": float(g_e.sum()),
        "shared_converter_violation_kw": float(max(0.0, np.max(pv_e + g_e + e_u - up))),
        "monthly_demand_upper_ok": bool(np.all(d_implied <= base_d + 1e-6)),
        "self_use_ratio": self_use,
        "self_use_ok": bool(
            float(pv_gen.sum()) <= 1e-9 or self_use >= cfg.MIN_SELF_USE - 1e-6
        ),
        "user_cost_reconstruction_error": abs(user_re - user_model),
        "cashflow_reconstruction_error": abs(cf1_re - cf1_model),
        "grid_charging_cost_yuan": charging_cost_re,
        "grid_charging_cost_reconstruction_error": abs(
            charging_cost_re - charging_cost_model
        ),
        "pricing_rule": pricing["pricing_rule"],
        "pv_settlement_revenue": float(pricing["pv_settlement"]),
        "ess_settlement_revenue": float(pricing["ess_settlement"]),
        "arbitrage_saving": 0.0,
        "arbitrage_service_fee": 0.0,
        "arbitrage_saving_nonnegative": True,
        "entry_state_consistent": entry_consistent,
        "fixed_cost_binding_ok": fixed_cost_ok,
    }


def solve_planning(
    alpha: float,
    gamma: float,
    *,
    pricing_rule: str | None = None,
    arbitrage_share: float | None = None,
    discount_rate: float | None = None,
    tou_multiplier: float = 1.0,
    include_user: bool = True,
    require_build: bool = False,
    investment_state: str = "free",
    threads: int | None = None,
    allow_uncertified: bool = False,
    time_limit: float | None = None,
) -> dict:
    """事前弹性求解：容量可重优化，15 年 NPV 目标。

    代表年按第 1 年满容量（degrade=1.0）求解，第 2~15 年由退化系数
    k=(1-DEGRADE)^(y-1) 折减组装（电池流按 k 缩放、缺口转上网/电网购电），
    退化只计一次。require_build=True 时强制建设（y_PV=1）：用于边界探针取
    V_build*（建设状态最优值，不含"不投资=0"选项），支持六状态分类。

    ``include_user=True`` 为默认：用户总成本约束（用户不吃亏）是合同硬
    约束，规划、锁定评估与边界探针共用同一约束集。
    """
    return _solve(
        alpha=alpha,
        gamma=gamma,
        pricing_rule=pricing_rule,
        arbitrage_share=arbitrage_share,
        discount_rate=discount_rate,
        tou_multiplier=tou_multiplier,
        capacity=None,
        degrade=1.0,
        include_user=include_user,
        annual_only=False,
        require_build=require_build,
        investment_state=investment_state,
        threads=threads,
        allow_uncertified=allow_uncertified,
        time_limit=time_limit,
    )


def solve_locked(
    alpha: float,
    gamma: float,
    capacity: tuple[float, float],
    *,
    pricing_rule: str | None = None,
    arbitrage_share: float | None = None,
    discount_rate: float | None = None,
    tou_multiplier: float = 1.0,
    include_user: bool = True,
    enforce_self_use: bool = False,
    export_max_kw: float | None = None,
    threads: int | None = None,
    allow_uncertified: bool = False,
    time_limit: float | None = None,
) -> dict:
    """固定容量运行评估：不重新优化，可选施加消纳率与并网上限。

    ``include_user=True`` 为默认且不可默认放松：用户总成本不超过
    ``LIFE × alpha × 无项目成本``（用户不吃亏）是合同硬约束，签约前
    规划、容量锁定后的运行评估和边界探针必须在同一约束集下比较，
    否则"锁定容量价值"会因缺少该约束而虚高、与规划最优值不可比。
    该约束只限制投资方从用户侧能收多少；运行期消纳率和并网上限由参数控制。
    """
    if len(capacity) != 2:
        raise ValueError("capacity 必须为 (pv_kw, ess_kwh)")
    pv_kw, ess_kwh = (float(value) for value in capacity)
    if not all(np.isfinite(value) and value >= 0.0 for value in (pv_kw, ess_kwh)):
        raise ValueError("固定容量必须为非负有限数")
    return _solve(
        alpha=alpha,
        gamma=gamma,
        pricing_rule=pricing_rule,
        arbitrage_share=arbitrage_share,
        discount_rate=discount_rate,
        tou_multiplier=tou_multiplier,
        capacity=(pv_kw, ess_kwh),
        degrade=1.0,
        include_user=include_user,
        annual_only=False,
        enforce_self_use=enforce_self_use,
        export_max_kw=export_max_kw,
        threads=threads,
        allow_uncertified=allow_uncertified,
        time_limit=time_limit,
    )
