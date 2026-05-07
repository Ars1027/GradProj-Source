"""
=====================

中文说明
--------

PF + metrics -> Runners -> Plotting -> Problem parsing -> main()。

This version extends benchmark_moivy_v2.py with:

1) Harder benchmark suites (DTLZ3/4/7, WFG1-9, ZDT3/4/6, etc.)
2) Parameterized problem specs from CLI: e.g. "wfg4:3:24" or "dtlz3:2:30"
3) Plotting for 2 objectives (2D) and 3 objectives (3D + 2D projections)

Why you want this:
- ZDT1/ZDT2/DTLZ2(2obj) are relatively easy; many algorithms converge similarly.
- WFG and DTLZ3/4/7 introduce multimodality, deception, bias, disconnected PFs.

Run examples:

  # Hard bi-objective suite (still shows nice 2D true PF overlay)
  python benchmark_moivy_v3.py --suite hard2 --pop 100 --gen 400 --seeds 1 2 3 4 5 --archive 300 --outdir results_hard2

  # Hard tri-objective suite (3D true PF overlay)
  python benchmark_moivy_v3.py --suite hard3 --pop 120 --gen 500 --seeds 1 2 3 --archive 400 --outdir results_hard3

  # Custom list ("name[:n_obj:n_var[:k[:l]]]" for WFG)
  python benchmark_moivy_v3.py --problems zdt4 zdt6 dtlz3:2:30 wfg4:2:24 wfg9:2:24 --pop 120 --gen 400 --outdir results_custom

Outputs:
- Same folder structure as v2, but problem folder uses a unique "problem_id" so
  different parameterizations won't overwrite each other.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- robust imports across pymoo versions ---
try:
    from pymoo.problems import get_problem
except Exception:  # pragma: no cover
    from pymoo.factory import get_problem  # type: ignore

from pymoo.optimize import minimize
from pymoo.termination import get_termination

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus

from moivy_v2 import moivy_optimize


# =============================================================================
# Styling (no extra deps)
# =============================================================================


# -----------------------------------------------------------------------------
# 绘图风格设置
# 尝试使用更易读的 matplotlib 风格；如果当前环境没有对应样式，就继续回退。
# -----------------------------------------------------------------------------
def apply_plot_style():
    """Use a nicer matplotlib style if available."""

    for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(s)
            return
        except Exception:
            pass


# =============================================================================
# PF + metrics
# =============================================================================


# -----------------------------------------------------------------------------
# 获取参考 Pareto Front
# 这些参考点会被用于计算 IGD / HV / Delta 等指标，也会用于画图对比。
# -----------------------------------------------------------------------------
def pareto_front(problem, n_points: int = 3000) -> np.ndarray:
    """Get a reference PF sample from a pymoo problem.

    Prefer requesting a specific number of points. Falls back to default.
    """

    for kwargs in (
        {"n_pareto_points": n_points},
        {"n_points": n_points},
        {},
    ):
        try:
            pf = problem.pareto_front(**kwargs)  # type: ignore
            if pf is not None and len(pf) > 0:
                return np.asarray(pf, dtype=float)
        except Exception:
            continue

    raise RuntimeError(
        f"Problem '{getattr(problem, 'name', lambda: '?')()}' does not provide a Pareto front sample. "
        "Choose another problem (DTLZ/WFG/ZDT usually work) or provide your own PF sample."
    )


# 保证输入前沿只保留非支配点，避免重复或被支配点干扰后续指标计算。
def ensure_nondominated(F: np.ndarray) -> np.ndarray:
    if F is None or len(F) == 0:
        return np.zeros((0, 0))
    F = np.asarray(F, dtype=float)
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[nd]


# 加性 epsilon 指标：衡量解集 A 至少需要整体平移多少，才能弱支配参考前沿 Z。
def epsilon_additive(A: np.ndarray, Z: np.ndarray) -> float:
    """Additive epsilon indicator for minimization."""
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if A.size == 0 or Z.size == 0:
        return float("nan")

    diff = A[None, :, :] - Z[:, None, :]
    d = np.max(diff, axis=2)
    d = np.min(d, axis=1)
    return float(np.max(d))


# Spacing 指标：看最终解集中各点间距是否均匀，越小说明分布越均匀。
def spacing(F: np.ndarray) -> float:
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n <= 2:
        return 0.0
    D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    d = D.min(axis=1)
    return float(np.std(d))


# 2 目标下的 Deb spread (Δ)，同时考虑两端覆盖与中间间隔均匀性。
def delta_spread_2d(F: np.ndarray, pf: np.ndarray) -> float:
    """Deb's spread (Δ) only defined for 2 objectives."""
    F = np.asarray(F, dtype=float)
    pf = np.asarray(pf, dtype=float)
    if F.shape[0] < 2 or F.shape[1] != 2:
        return float("nan")

    idx = np.argsort(F[:, 0], kind="mergesort")
    Fs = F[idx]
    d = np.linalg.norm(Fs[1:] - Fs[:-1], axis=1)
    d_bar = np.mean(d) if len(d) > 0 else 0.0

    pf_idx = np.argsort(pf[:, 0], kind="mergesort")
    pf_s = pf[pf_idx]
    df = np.linalg.norm(Fs[0] - pf_s[0])
    dl = np.linalg.norm(Fs[-1] - pf_s[-1])
    if d_bar == 0:
        return float("nan")
    return float((df + dl + np.sum(np.abs(d - d_bar))) / (df + dl + len(d) * d_bar))


# 判断 a 是否支配 b（默认最小化问题）。
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


# Coverage C(A,B)：B 中有多少比例的点被 A 中至少一个点支配。
def coverage(A: np.ndarray, B: np.ndarray) -> float:
    A = ensure_nondominated(A)
    B = ensure_nondominated(B)
    if B.size == 0:
        return float("nan")
    if A.size == 0:
        return 0.0

    le = (A[:, None, :] <= B[None, :, :]).all(axis=2)
    lt = (A[:, None, :] < B[None, :, :]).any(axis=2)
    dom = le & lt
    return float(dom.any(axis=0).mean())


# 统一构造各类指标对象，避免在每次调用时重复初始化。
def build_indicators(pf: np.ndarray, hv_ref: np.ndarray):
    return {
        "GD": GD(pf),
        "GDPlus": GDPlus(pf),
        "IGD": IGD(pf),
        "IGDPlus": IGDPlus(pf),
        "HV": HV(ref_point=hv_ref),
    }


# 计算最终评价时使用的完整指标集。
def calc_all_metrics(F: np.ndarray, pf: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, float]:
    F = ensure_nondominated(F)
    out: Dict[str, float] = {}
    out["GD"] = float(indicators["GD"](F))
    out["GDPlus"] = float(indicators["GDPlus"](F))
    out["IGD"] = float(indicators["IGD"](F))
    out["IGDPlus"] = float(indicators["IGDPlus"](F))
    out["HV"] = float(indicators["HV"](F))
    out["EpsilonAdd"] = epsilon_additive(F, pf)
    out["Spacing"] = spacing(F)
    out["Delta"] = delta_spread_2d(F, pf)
    out["N_NDS"] = float(F.shape[0])
    return out


# 计算收敛过程里常用的核心指标，避免每代都算太多指标导致开销过大。
def calc_core_metrics(F: np.ndarray, indicators: Dict[str, Any]) -> Dict[str, float]:
    F = ensure_nondominated(F)
    return {
        "IGD": float(indicators["IGD"](F)),
        "IGDPlus": float(indicators["IGDPlus"](F)),
        "HV": float(indicators["HV"](F)),
        "N_NDS": float(F.shape[0]),
    }


# =============================================================================
# Runners
# =============================================================================


# 单次实验的统一返回结构。
@dataclass
class RunResult:
    F_final: np.ndarray
    X_final: Optional[np.ndarray]
    runtime_sec: float
    n_eval: int
    history: List[Dict[str, Any]]


# pymoo 算法的回调记录器：每一代抓取当前最优/当前种群前沿并保存指标。
class PymooRecorder:
    def __init__(self, indicators: Dict[str, Any]):
        self.indicators = indicators
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, algorithm):
        gen = int(getattr(algorithm, "n_gen", len(self.rows) + 1))
        F = None
        try:
            if algorithm.opt is not None:
                F = algorithm.opt.get("F")
        except Exception:
            F = None
        if F is None:
            try:
                F = algorithm.pop.get("F")
            except Exception:
                F = None
        if F is None:
            return
        m = calc_core_metrics(F, self.indicators)
        m["gen"] = gen
        self.rows.append(m)


# 运行一次 pymoo 自带算法（如 NSGA-II / MOEA-D / SPEA2）。
def run_one_pymoo(problem, algorithm, n_gen: int, seed: int, indicators: Dict[str, Any]) -> RunResult:
    rec = PymooRecorder(indicators)
    t0 = time.perf_counter()
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
        callback=rec,
    )
    t1 = time.perf_counter()
    F_final = np.asarray(res.opt.get("F"), dtype=float)
    X_final = res.opt.get("X")
    X_final = np.asarray(X_final, dtype=float) if X_final is not None else None
    n_eval = int(getattr(res.algorithm.evaluator, "n_eval", 0))
    return RunResult(F_final=F_final, X_final=X_final, runtime_sec=float(t1 - t0), n_eval=n_eval, history=rec.rows)


# 运行一次HMOIVY，同时通过 callback 记录每代 archive 的指标。
def run_one_moivy(problem, pop_size: int, n_gen: int, seed: int, archive_size: int, indicators: Dict[str, Any]) -> RunResult:
    rows: List[Dict[str, Any]] = []

    def cb(info: Dict[str, Any]):
        gen = int(info["gen"])
        F = info["archive_F"]
        m = calc_core_metrics(F, indicators)
        m["gen"] = gen
        rows.append(m)

    t0 = time.perf_counter()
    res = moivy_optimize(
        problem,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        archive_size=archive_size,
        callback=cb,
        verbose=False,
    )
    t1 = time.perf_counter()

    F_final = np.asarray(res.archive_F, dtype=float)
    X_final = np.asarray(res.archive_X, dtype=float)
    n_eval = int(pop_size + n_gen * pop_size)
    return RunResult(F_final=F_final, X_final=X_final, runtime_sec=float(t1 - t0), n_eval=n_eval, history=rows)


# =============================================================================
# Plotting
# =============================================================================


# 保存最终前沿，便于后续单独分析或复画图。
def save_front_npz(path: Path, F: np.ndarray, X: Optional[np.ndarray], pf: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if X is None:
        np.savez_compressed(path, F=F, pf=pf)
    else:
        np.savez_compressed(path, F=F, X=X, pf=pf)


# 2D 前沿叠加图：把真实 PF 与各算法最终 PF 画在同一张图上。
def plot_overlay_2d(out_png: Path, out_pdf: Path, pf: np.ndarray, fronts: Dict[str, np.ndarray], title: str):
    apply_plot_style()
    plt.figure(figsize=(8.0, 6.2), dpi=220)

    pf = np.asarray(pf, dtype=float)
    pf_sorted = pf[np.argsort(pf[:, 0], kind="mergesort")]
    plt.plot(pf_sorted[:, 0], pf_sorted[:, 1], linewidth=2.8, label="True PF")

    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    for i, (name, F) in enumerate(fronts.items()):
        F = ensure_nondominated(F)
        if F.size == 0:
            continue
        plt.scatter(F[:, 0], F[:, 1], s=26, marker=markers[i % len(markers)], alpha=0.85, label=name)

    plt.title(title)
    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# 3D 前沿叠加图：用于三目标问题的可视化对比。
def plot_overlay_3d(out_png: Path, out_pdf: Path, pf: np.ndarray, fronts: Dict[str, np.ndarray], title: str):
    apply_plot_style()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8.4, 6.6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    pf = np.asarray(pf, dtype=float)
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], s=6, alpha=0.25, label="True PF")

    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    for i, (name, F) in enumerate(fronts.items()):
        F = ensure_nondominated(F)
        if F.size == 0:
            continue
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=26, marker=markers[i % len(markers)], alpha=0.85, label=name)

    ax.set_title(title)
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_zlabel("$f_3$")
    ax.view_init(elev=22, azim=40)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# 3 目标的三组二维投影图
def plot_projections_3d(out_dir: Path, pf: np.ndarray, fronts: Dict[str, np.ndarray], title_prefix: str):
    """For 3 objectives, also generate (f1,f2), (f1,f3), (f2,f3) projections."""
    pairs = [(0, 1), (0, 2), (1, 2)]
    for (i, j) in pairs:
        apply_plot_style()
        plt.figure(figsize=(8.0, 6.2), dpi=220)
        pf2 = pf[:, [i, j]]
        plt.scatter(pf2[:, 0], pf2[:, 1], s=8, alpha=0.25, label="True PF")
        markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
        for k, (name, F) in enumerate(fronts.items()):
            F = ensure_nondominated(F)
            if F.size == 0:
                continue
            Fj = F[:, [i, j]]
            plt.scatter(Fj[:, 0], Fj[:, 1], s=26, marker=markers[k % len(markers)], alpha=0.85, label=name)
        plt.title(f"{title_prefix} | Projection (f{i+1}, f{j+1})")
        plt.xlabel(f"$f_{i+1}$")
        plt.ylabel(f"$f_{j+1}$")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"pareto_proj_f{i+1}_f{j+1}.png", bbox_inches="tight")
        plt.savefig(out_dir / f"pareto_proj_f{i+1}_f{j+1}.pdf", bbox_inches="tight")
        plt.close()


# 收敛曲线：按代数聚合多次 seed 的均值和标准差。
def plot_convergence(out_png: Path, out_pdf: Path, df_conv: pd.DataFrame, metric: str, title: str):
    apply_plot_style()
    plt.figure(figsize=(8.0, 6.0), dpi=220)

    for alg in df_conv["algorithm"].unique().tolist():
        sub = df_conv[df_conv["algorithm"] == alg]
        g = sub.groupby("gen")[metric].agg(["mean", "std"]).reset_index()
        g["std"] = g["std"].fillna(0.0)
        plt.plot(g["gen"], g["mean"], linewidth=2.2, label=alg)
        plt.fill_between(g["gen"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# coverage 热力图：用矩阵方式展示 C(A,B) 的均值，比较各算法相互支配能力。
def plot_coverage_heatmap(out_png: Path, out_pdf: Path, cov_mean: pd.DataFrame, title: str):
    apply_plot_style()
    plt.figure(figsize=(8.0, 6.4), dpi=220)
    M = cov_mean.values.astype(float)
    im = plt.imshow(M, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(cov_mean.shape[1]), cov_mean.columns, rotation=45, ha="right")
    plt.yticks(range(cov_mean.shape[0]), cov_mean.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            s = "NA" if np.isnan(v) else f"{v:.2f}"
            plt.text(j, i, s, ha="center", va="center")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


# =============================================================================
# Problem selection
# =============================================================================


SUITES: Dict[str, List[str]] = {
    # Baseline easy
    "easy": ["zdt1", "zdt2", "dtlz2:2:12"],

    # Harder but still 2 objectives (keeps 2D plotting)
    # - ZDT4 (multimodal), ZDT6 (non-uniform density), ZDT3 (disconnected)
    # - DTLZ3/4/7 with 2 objectives but larger n_var => harder landscape
    # - WFG4/5/9 (multimodal/deceptive/parameter dependencies)
    "hard2": [
        "zdt3",
        "zdt4",
        "zdt6",
        "dtlz3:2:12",
        "dtlz4:2:30",
        # "dtlz7:2:30",
        "wfg4:2:24",
        "wfg5:2:24",
        "wfg9:2:24",
    ],

    # Hard 3-objective suite (3D plot)
    "hard3": [
        "dtlz3:3:12",
        "dtlz4:3:30",
        # "dtlz7:3:30",
        "wfg4:3:24",
        "wfg9:3:24",
    ],
}


# 解析命令行里的问题描述，例如：wfg4:3:24 或 dtlz3:2:30。
def parse_problem_spec(spec: str):
    """Parse a problem spec string.

    Supported formats:
      - name
      - name:n_obj:n_var
      - wfgX:n_obj:n_var[:k[:l]]

    Returns (problem, problem_id)
    """

    parts = spec.strip().split(":")
    name = parts[0].lower()

    kwargs: Dict[str, Any] = {}
    if len(parts) >= 3:
        kwargs["n_obj"] = int(parts[1])
        kwargs["n_var"] = int(parts[2])

    if name.startswith("wfg"):
        if len(parts) >= 4:
            kwargs["k"] = int(parts[3])
        if len(parts) >= 5:
            kwargs["l"] = int(parts[4])

    problem = get_problem(name, **kwargs) if kwargs else get_problem(name)

    # unique id for output folders
    pid = name
    if "n_obj" in kwargs:
        pid += f"_M{kwargs['n_obj']}"
    if "n_var" in kwargs:
        pid += f"_V{kwargs['n_var']}"
    if "k" in kwargs:
        pid += f"_k{kwargs['k']}"
    if "l" in kwargs:
        pid += f"_l{kwargs['l']}"

    return problem, pid


# 给 MOEA/D 构造参考方向，数量尽量接近种群规模。
def choose_ref_dirs(pop_size: int, n_obj: int) -> np.ndarray:
    """Choose a reference direction set with size close to pop_size."""

    if n_obj == 2:
        return get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)

    # Increase partitions until we get >= pop_size directions
    p = 1
    while True:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=p)
        if len(ref_dirs) >= pop_size or p >= 50:
            return ref_dirs
        p += 1


# =============================================================================
# Main
# =============================================================================

def main():
    # ---------- 解析命令行参数 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=sorted(SUITES.keys()),
        help=f"Use a predefined suite: {sorted(SUITES.keys())}. If set, --problems is ignored.",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=["zdt1", "zdt2", "dtlz2:2:12"],
        help="Problem list. Use 'name[:n_obj:n_var[:k[:l]]]' for parameterized problems (WFG/DTLZ).",
    )
    parser.add_argument("--pop", type=int, default=80)
    parser.add_argument("--gen", type=int, default=200)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--archive", type=int, default=250)
    parser.add_argument("--outdir", type=str, default="results_moivy")
    parser.add_argument("--pf_points", type=int, default=4000, help="How many PF samples to request (if supported).")
    args = parser.parse_args()

    # ---------- 读取并整理运行配置 ----------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pop_size = int(args.pop)
    n_gen = int(args.gen)
    seeds = [int(s) for s in args.seeds]
    archive_size = int(args.archive)
    pf_points = int(args.pf_points)

    # 如果指定了 suite，就用预定义问题集；否则按用户传入的问题列表运行。
    problem_specs = SUITES[args.suite] if args.suite else args.problems

    all_detail_rows: List[Dict[str, Any]] = []
    all_conv_rows: List[Dict[str, Any]] = []

    # ==================== 逐问题运行实验 ====================
    for spec in problem_specs:
        # 创建具体测试问题，并生成唯一 problem_id 作为输出目录名。
        problem, pid = parse_problem_spec(spec)
        pf = pareto_front(problem, n_points=pf_points)

        # HV 参考点通常选在 PF 的“外侧”，这里简单取各维最大值再加一个安全边距。
        hv_ref = pf.max(axis=0) + 0.1
        indicators = build_indicators(pf, hv_ref)

        ref_dirs = choose_ref_dirs(pop_size, problem.n_obj)

        # 统一管理待比较算法。HMOIVY 单独调用，其余算法通过工厂函数延迟创建。
        algs = {
            "HMOIVY": None,
            "NSGA-II": lambda: NSGA2(pop_size=pop_size),
            "MOEA/D": lambda: MOEAD(ref_dirs=ref_dirs, n_neighbors=min(20, len(ref_dirs))),
            "SPEA2": lambda: SPEA2(pop_size=pop_size),
        }

        fronts_by_alg_seed: Dict[Tuple[str, int], np.ndarray] = {}

        # -------------------- 多随机种子重复实验 --------------------
        for seed in seeds:
            # 先跑HMOIVY，并记录最终指标、每代指标和最终前沿。
            rr = run_one_moivy(problem, pop_size, n_gen, seed, archive_size, indicators)
            m_final = calc_all_metrics(rr.F_final, pf, indicators)
            m_final.update(
                {
                    "problem": pid,
                    "algorithm": "HMOIVY",
                    "seed": seed,
                    "runtime_sec": rr.runtime_sec,
                    "n_eval": rr.n_eval,
                }
            )
            all_detail_rows.append(m_final)
            fronts_by_alg_seed[("HMOIVY", seed)] = rr.F_final

            for row in rr.history:
                r = dict(row)
                r["problem"] = pid
                r["algorithm"] = "HMOIVY"
                r["seed"] = seed
                all_conv_rows.append(r)

            save_front_npz(
                outdir / pid / "fronts" / f"front_HMOIVY_seed{seed}.npz",
                ensure_nondominated(rr.F_final),
                rr.X_final,
                pf,
            )

            # 再跑其余基线算法，保持同样的代数设置，便于公平比较。
            for alg_name, factory in algs.items():
                if alg_name == "HMOIVY":
                    continue
                alg = factory()
                rr2 = run_one_pymoo(problem, alg, n_gen=n_gen, seed=seed, indicators=indicators)
                m_final2 = calc_all_metrics(rr2.F_final, pf, indicators)
                m_final2.update(
                    {
                        "problem": pid,
                        "algorithm": alg_name,
                        "seed": seed,
                        "runtime_sec": rr2.runtime_sec,
                        "n_eval": rr2.n_eval,
                    }
                )
                all_detail_rows.append(m_final2)
                fronts_by_alg_seed[(alg_name, seed)] = rr2.F_final

                for row in rr2.history:
                    r = dict(row)
                    r["problem"] = pid
                    r["algorithm"] = alg_name
                    r["seed"] = seed
                    all_conv_rows.append(r)

                save_front_npz(
                    outdir / pid / "fronts" / f"front_{alg_name.replace('/','_').replace(' ','')}_seed{seed}.npz",
                    ensure_nondominated(rr2.F_final),
                    rr2.X_final,
                    pf,
                )

        # ---------- 保存当前问题的详细结果与收敛记录 ----------
        df_detail = pd.DataFrame([r for r in all_detail_rows if r["problem"] == pid])
        df_conv = pd.DataFrame([r for r in all_conv_rows if r["problem"] == pid])
        (outdir / pid).mkdir(parents=True, exist_ok=True)
        df_detail.to_csv(outdir / pid / "details.csv", index=False)
        df_conv.to_csv(outdir / pid / "convergence.csv", index=False)

        # ---------- 计算 coverage 矩阵并绘制热力图 ----------
        alg_names = list(algs.keys())
        cov_rows = []
        # -------------------- 多随机种子重复实验 --------------------
        for seed in seeds:
            for A in alg_names:
                for B in alg_names:
                    if A == B:
                        continue
                    FA = fronts_by_alg_seed.get((A, seed))
                    FB = fronts_by_alg_seed.get((B, seed))
                    c = float("nan") if (FA is None or FB is None) else coverage(FA, FB)
                    cov_rows.append({"problem": pid, "seed": seed, "A": A, "B": B, "C": c})
        df_cov = pd.DataFrame(cov_rows)
        df_cov.to_csv(outdir / pid / "coverage.csv", index=False)
        cov_mean = df_cov.groupby(["A", "B"])["C"].mean().unstack("B").reindex(index=alg_names, columns=alg_names)
        for a in alg_names:
            cov_mean.loc[a, a] = np.nan
        cov_mean.to_csv(outdir / pid / "coverage_mean.csv")
        plot_coverage_heatmap(
            outdir / pid / "plots" / "coverage_heatmap.png",
            outdir / pid / "plots" / "coverage_heatmap.pdf",
            cov_mean,
            title=f"{pid} | Coverage C(A,B) mean over seeds",
        )

        # ---------- 画最终前沿对比图 ----------
        fixed_seed = int(seeds[0])
        seed_fronts: Dict[str, np.ndarray] = {a: fronts_by_alg_seed[(a, fixed_seed)] for a in alg_names if (a, fixed_seed) in fronts_by_alg_seed}

        best_fronts: Dict[str, np.ndarray] = {}
        for alg_name in alg_names:
            sub = df_detail[df_detail["algorithm"] == alg_name]
            if len(sub) == 0:
                continue
            best = sub.sort_values(["HV", "IGD"], ascending=[False, True]).iloc[0]
            best_seed = int(best["seed"])
            best_fronts[alg_name] = fronts_by_alg_seed[(alg_name, best_seed)]

        m = int(problem.n_obj)
        if m == 2:
            plot_overlay_2d(
                outdir / pid / "plots" / f"pareto_overlay_seed{fixed_seed}.png",
                outdir / pid / "plots" / f"pareto_overlay_seed{fixed_seed}.pdf",
                pf,
                seed_fronts,
                title=f"{pid} | True PF vs Final PF (seed={fixed_seed})",
            )
            plot_overlay_2d(
                outdir / pid / "plots" / "pareto_overlay_best.png",
                outdir / pid / "plots" / "pareto_overlay_best.pdf",
                pf,
                best_fronts,
                title=f"{pid} | True PF vs Final PF (best seed by HV)",
            )

        elif m == 3:
            plot_overlay_3d(
                outdir / pid / "plots" / f"pareto_overlay_3d_seed{fixed_seed}.png",
                outdir / pid / "plots" / f"pareto_overlay_3d_seed{fixed_seed}.pdf",
                pf,
                seed_fronts,
                title=f"{pid} | True PF vs Final PF (seed={fixed_seed})",
            )
            plot_overlay_3d(
                outdir / pid / "plots" / "pareto_overlay_3d_best.png",
                outdir / pid / "plots" / "pareto_overlay_3d_best.pdf",
                pf,
                best_fronts,
                title=f"{pid} | True PF vs Final PF (best seed by HV)",
            )

            # extra 2D projections
            plot_projections_3d(outdir / pid / "plots" / "projections_seed", pf, seed_fronts, title_prefix=f"{pid} (seed={fixed_seed})")
            plot_projections_3d(outdir / pid / "plots" / "projections_best", pf, best_fronts, title_prefix=f"{pid} (best-by-HV)")

        # ---------- 画收敛曲线（HV / IGD / IGDPlus） ----------
        for metric in ["HV", "IGD", "IGDPlus"]:
            if metric not in df_conv.columns:
                continue
            plot_convergence(
                outdir / pid / "plots" / f"conv_{metric}.png",
                outdir / pid / "plots" / f"conv_{metric}.pdf",
                df_conv,
                metric=metric,
                title=f"{pid} | Convergence ({metric}) mean±std over seeds",
            )

    # ==================== 全部问题汇总 ====================
    df_all = pd.DataFrame(all_detail_rows)
    df_all.to_csv(outdir / "details_all.csv", index=False)

    metrics_cols = [
        "GD",
        "GDPlus",
        "IGD",
        "IGDPlus",
        "HV",
        "EpsilonAdd",
        "Spacing",
        "Delta",
        "N_NDS",
        "runtime_sec",
        "n_eval",
    ]
    agg_rows = []
    for (prob, alg), sub in df_all.groupby(["problem", "algorithm"]):
        row = {"problem": prob, "algorithm": alg}
        for c in metrics_cols:
            if c in sub.columns:
                row[c + "_mean"] = float(sub[c].mean())
                row[c + "_std"] = float(sub[c].std(ddof=1)) if len(sub) > 1 else 0.0
        agg_rows.append(row)
    df_sum = pd.DataFrame(agg_rows).sort_values(["problem", "IGD_mean"], ascending=[True, True])
    df_sum.to_csv(outdir / "summary.csv", index=False)
    print("\n=== Summary saved to:", outdir / "summary.csv")
    print(df_sum.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
