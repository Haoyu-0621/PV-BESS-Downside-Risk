"""参数配置：全部输入口径集中在单文件，便于核验与敏感性覆盖。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- 数据文件 ----
INPUT_DIR = ROOT / "inputs"
RAW_LOAD_FILE = INPUT_DIR / "luoyuan_load.xlsx"
PV_PROFILE_FILE = INPUT_DIR / "luoyuan_pv_output.xlsx"
PV_REF_FILE = INPUT_DIR / "pv_reference_capacity.xlsx"
PV_REF_NAME = "Luoyuan Service Area"
TOU_FILE = INPUT_DIR / "fuzhou_tou_tariff.xlsx"

N = 8760
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# ---- 容量 ----
PV_MIN = 0.0  # kW；光伏容量允许取 0
PV_MAX = 3000.0  # kW
ESS_MIN = 0.0  # kWh；储能能量容量允许取 0
ESS_MAX = 2000.0  # kWh
ESS_ENTRY_POWER_KW = 0.0  # kW；储能可不安装；正值表示安装后的最低功率
C_RATE = 0.5  # P_ESS = 0.5 × E_ESS
SOC_LO, SOC_HI = 0.05, 0.95
ROUNDTRIP = 0.90
DEGRADE = 0.022  # 2.2%/年（时间衰减假设，待核验）
PV_DEGRADE = 0.005  # 光伏年衰减：0.5%/年（常用基准假设）
MIN_SELF_USE = 0.70  # 年度最低光伏消纳率（规划期"上网代理约束"，仅规划求解强制）

# ---- 上网功率上限（并网容量，物理约束，规划与锁定均适用）----
# None = 不限制（2026-09 用户决定：不知道用户变压器/并网容量，不主观设定上网上限；
# 光伏上网的"代理限制"由 MIN_SELF_USE 消纳率承担）。
EXPORT_MAX_KW: float | None = None  # 并网容量（上网功率上限，kW）；None = 不限制
# 锁定容量压力状态的并网容量；基准规划与锁定基准状态仍使用 EXPORT_MAX_KW。
LOCKED_EXPORT_MAX_KW = 1000.0

# ---- 工程可行集（进入/筛选求解专用，见《实验方案设计.md》1.3）----
# 半连续投资状态：光伏 ∈ {0} ∪ [PV_MIN_ENTRY_KW, PV_MAX]，
#                 储能 ∈ {0} ∪ [ESS_MIN_ENTRY_KWH, ESS_MAX]。
# 固定工程费已删除（2026-08 老师确认：实际工程费难以核实，不设置）。
PV_MIN_ENTRY_KW = 200.0  # kW；最小可建光伏（基准假设，需补工程/商业出处 + 门槛敏感性）
ESS_MIN_ENTRY_KWH = 100.0  # kWh；最小可建储能（0.5C 派生 50 kW，非独立功率下界；储能进入功率参数保持 0）
ALLOW_ESS_ONLY = False  # False 时强制 y_ESS <= y_PV（禁止 ESS-only）；合同要求光储一体时另设 y_ESS = y_PV

# ---- 经济 ----
PV_CAPEX = 3800.0  # 元/kW
ESS_CAPEX = 1000.0  # 元/kWh
FIXED_ENG = 0.0  # 固定工程费（元）；2026-08 定稿删除（投资=容量线性成本）。=0 保持该口径；>0 作敏感性/对照（一次性入场费压低最优容量）
PV_OM = 40.0  # 元/kW/年
ESS_OM_RATE = 0.02
VAR_OM = 0.0
DEMAND_CHARGE = 40.0  # 元/kW/月（按最大需量计费；需量 = 月最大电功率，非变压器容量）
DEMAND_PEAK = 1.15  # 需量裕度/安全系数（工程经验值，1.1~1.2 合理，2026-09 用户确认保留，待核验）
DEMAND_SHARE = 0.2  # 投资方分享用户需量节省的比例：削峰节省先落在用户电费单，用户按此比例
#                    支付投资方作为削峰收益（savings-share EMC 惯例，合同参数待核验）；用户保留 80%
# 基准合同价格系数：0.8 表示按目录 TOU 的 8 折结算（待核验）。
# gamma 本身就是相对 TOU 的绝对价格系数，模型中不再写成
# gamma * SELL_DISCOUNT * tou；基准点为 (alpha, gamma)=(1.0, BASELINE_GAMMA)。
BASELINE_GAMMA = 0.80
# 兼容归档脚本和旧输出读取逻辑；有效模型不再将其作为隐藏乘数使用。
SELL_DISCOUNT = BASELINE_GAMMA
EXPORT_PRICE = 0.35  # 元/kWh；余电上网（福建燃煤基准价量级）
R_REQ = 0.05  # 要求折现率（待财务依据核验）
SALVAGE = 0.05
LIFE = 15
ESS_FEE_KWH = 0.0  # 储能服务费：EMC主模型暂不另收，价值通过项目结算回收

# ---- 结算规则 ----
PRICING_RULE_TOU_SERVICE = "tou_storage_service"  # 主规则：逐时分时电价
PRICING_RULE = PRICING_RULE_TOU_SERVICE  # 当前默认采用分时电价规则
ARBITRAGE_SHARE = 0.0  # 套利分成：当前版本不设置；电网充电成本由项目内部核算承担

# ---- 求解 ----
THREADS = 4
VERBOSE = 0
TIME_LIMIT = 600.0  # 秒；单次求解时间上限（正常 ~90s，防止病态 MILP 卡死流水线）
# 结算口径（2026-09 EMC 统一绿电折扣）：用户承担负荷购电，投资方承担电网
# 给电池充电的购电成本；光伏直供和储能放电按分时电价折扣向用户结算。
# 电网充电成本直接从投资方现金流扣除，实际 NPV 与规划 NPV一致。
# 同时充放电必须被严格禁止（LP/MILP 都一样）：线性惩罚无法严格表达互斥，
# 故主口径采用二进制互斥（MUTEX=True）。
MUTEX = True  # 充放电互斥二进制：严格禁止同小时同时充放（主口径 MILP）
MUTEX_FORMULATION = "binary"  # binary 或 sos；SOS1 与二进制互斥可行域等价
PEN_SIMULT = 0.001  # 元/kWh；仅 MUTEX=False（LP 对照）时启用的软惩罚，主口径不用
MIP_GAP = 1e-3  # 相对间隙 0.1%
MIP_GAP_ABS = 1000.0  # 绝对间隙 1000 元：避免 NPV≈0 边界点相对间隙死磕
METHOD = 1  # 求解方法：1=对偶单纯形（E1 验证提速，LP 12s vs 默认 60s）
TOU_MIP_FOCUS = 1  # 新分时规则优先找可行 incumbent；旧基准规则不改变求解参数
TOU_METHOD = 0  # 新分时规则使用 Gurobi 自动 LP 方法；0=automatic，旧规则仍用 METHOD
TOU_LP_WARM_START = False  # 暂不执行连续松弛 warm-start；正式求解保持原始 MILP


def gamma_floor(base_sell: float) -> float:
    """价格地板：上网电价/基准售价。

    保留仅为"合同价等于上网价"的参考标注（γ_floor 处合同价 = 上网价，
    低于此价时投资方自用收益低于上网收益）。结算本身不再托底。
    """
    if base_sell <= 0:
        raise ValueError("基准售价必须为正数")
    return EXPORT_PRICE / base_sell


def sell_price(gamma: float, base_sell: float) -> float:
    """实际结算价：合同价 = γ×基准售价（无托底，可低于上网电价）。

    去掉 max(·, EXPORT_PRICE) 托底：合同价由谈判决定，用户要求降价时可
    低于上网价；此时投资方被迫按消纳率约束低价供绿电，NPV 可能转负 →
    不可投区（NPV<0）内生出现。
    """
    return gamma * base_sell


def invest_of(pv_kw: float, ess_kwh: float) -> float:
    return FIXED_ENG + PV_CAPEX * pv_kw + ESS_CAPEX * ess_kwh


# ---- 实验组织与文件输出 ----
# 工作产物集中在本项目的 codex工作文件；可通过 run.py --out 另行指定。
OUTPUT_ROOT = ROOT / "codex工作文件" / "实验结果"
LHS_N = 1000
LHS_SEED = 20260913
LHS_ALPHA_BOUNDS = (0.25, 1.0)
LHS_GAMMA_BOUNDS = (0.40, 1.0)
BASELINE_OUTPUT = OUTPUT_ROOT / "pv_priority_baseline"
LHS_OUTPUT = OUTPUT_ROOT / "locked_lhs_export_cap_1000_gamma1"
SENSITIVITY_OUTPUT = OUTPUT_ROOT / "sensitivity_reoptimized_with_fit"
# 正式结果中这两个难解情景采用等价 binary 互斥表达，合并后直接执行，免去手动补算。
SENSITIVITY_BINARY_CASES = ("demand_share_15pct", "discount_rate_6pct")
