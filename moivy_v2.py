from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance


# Utilities

def fast_non_dominated_sort_and_crowding(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    # -------------------------------------------------------------------------
    # 功能：
    #   对当前种群的目标值矩阵 F 做：
    #   1) 非支配排序（rank）
    #   2) 拥挤距离计算（crowding distance）
    #
    # -------------------------------------------------------------------------
    """
    Return:
      rank : (n,) int, 0 is first front
      crowd: (n,) float
      fronts: list of index arrays
    """
    nds = NonDominatedSorting()
    fronts = nds.do(F, return_rank=False)
    rank = np.empty(F.shape[0], dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    crowd = np.zeros(F.shape[0], dtype=float)
    for front in fronts:
        crowd[front] = calc_crowding_distance(F[front])

    return rank, crowd, fronts


def lhs_sample(
    rng: np.random.Generator, xl: np.ndarray, xu: np.ndarray, n_samples: int
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   使用简单的 Latin Hypercube Sampling（LHS）初始化种群。
    #   在每一维上分层采样，初始点分布通常更均匀，
    #   有助于提升早期搜索覆盖度，减少初始化扎堆。
    # -------------------------------------------------------------------------
    """
    Simple Latin Hypercube Sampling (no external deps).
    """
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    n_var = xl.size

    # (n_samples, n_var) in [0,1]
    U = rng.random((n_samples, n_var))
    X = np.empty_like(U)

    # stratified per dimension
    for j in range(n_var):
        perm = rng.permutation(n_samples)
        X[:, j] = (perm + U[:, j]) / n_samples

    return xl + X * (xu - xl)


def reflect_bounds(Y: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   对越界解进行“反射式”边界处理，而不是简单裁剪。
    #   如果某个变量冲出了边界，就像撞墙后弹回来，而不是直接卡在边界点。
    #   相比纯 clip，反射通常更能保留搜索方向信息，避免大量个体堆积在边界上。
    # -------------------------------------------------------------------------

    Y = np.asarray(Y, dtype=float, copy=True)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    # 广播到 (n, n_var) 形状（也支持标量边界）
    xl_b = np.broadcast_to(xl, Y.shape)
    xu_b = np.broadcast_to(xu, Y.shape)

    # 逐元素反射
    Y = np.where(Y < xl_b, 2.0 * xl_b - Y, Y)
    Y = np.where(Y > xu_b, 2.0 * xu_b - Y, Y)

    # 极端情况下仍可能越界，最后再 clip 一次
    np.clip(Y, xl, xu, out=Y)
    return Y



def polynomial_mutation(
    Y: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float = 20.0,
    prob: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   对实数编码个体执行多项式变异。

    #   eta 越小，变异步长通常越大，更偏探索；
    #   eta 越大，变异更细，更偏局部精修。
    #
    #   前期粗一些，后期细一些，先探索、后开发。
    # -------------------------------------------------------------------------
    """
    Vectorized polynomial mutation (Deb's PM) for real-coded variables.
    """
    if rng is None:
        rng = np.random.default_rng()

    Y = np.array(Y, dtype=float, copy=True)
    n, n_var = Y.shape

    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    if xl.ndim == 0:
        xl = np.full(n_var, xl)
    if xu.ndim == 0:
        xu = np.full(n_var, xu)

    prob = 1.0 / n_var if prob is None else float(prob)
    if prob <= 0:
        return Y

    xl_b = np.broadcast_to(xl[None, :], Y.shape)
    xu_b = np.broadcast_to(xu[None, :], Y.shape)
    rng_b = xu_b - xl_b
    rng_b = np.where(rng_b == 0, 1.0, rng_b)

    M = rng.random(size=Y.shape) < prob
    if not np.any(M):
        return Y

    delta1 = (Y - xl_b) / rng_b
    delta2 = (xu_b - Y) / rng_b
    rand = rng.random(size=Y.shape)
    mut_pow = 1.0 / (eta + 1.0)

    # mutate (rand <= 0.5)
    mask_low = M & (rand <= 0.5)
    if np.any(mask_low):
        xy = 1.0 - delta1[mask_low]
        val = 2.0 * rand[mask_low] + (1.0 - 2.0 * rand[mask_low]) * (xy ** (eta + 1.0))
        deltaq = val ** mut_pow - 1.0
        Y[mask_low] = Y[mask_low] + deltaq * rng_b[mask_low]

    # mutate (rand > 0.5)
    mask_high = M & (rand > 0.5)
    if np.any(mask_high):
        xy = 1.0 - delta2[mask_high]
        val = 2.0 * (1.0 - rand[mask_high]) + 2.0 * (rand[mask_high] - 0.5) * (xy ** (eta + 1.0))
        deltaq = 1.0 - val ** mut_pow
        Y[mask_high] = Y[mask_high] + deltaq * rng_b[mask_high]

    np.clip(Y, xl, xu, out=Y)
    return Y


def sbx_one_child(
    P1: np.ndarray,
    P2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   执行一个轻量化的交叉，并返回一个子代。
    # -------------------------------------------------------------------------
    """
    A lightweight SBX that returns one child (vectorized for batches).
    P1, P2: (n, n_var)
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)
    n, n_var = P1.shape

    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    # Uniform random for SBX
    u = rng.random((n, n_var))
    beta = np.empty_like(u)

    beta[u <= 0.5] = (2.0 * u[u <= 0.5]) ** (1.0 / (eta + 1.0))
    beta[u > 0.5] = (1.0 / (2.0 * (1.0 - u[u > 0.5]))) ** (1.0 / (eta + 1.0))

    # random sign
    s = rng.random((n, n_var)) < 0.5
    beta[s] = -beta[s]

    C = 0.5 * ((1.0 + beta) * P1 + (1.0 - beta) * P2)

    np.clip(C, xl, xu, out=C)
    return C


# =============================================================================
# Archive maintenance
# =============================================================================
# 这一部分负责外部存档 archive的维护。

def _hv2d_contributions(F: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   计算二维非支配解集中各点对 HV 的近似贡献。

    #   当 archive 太大时，需要删掉一部分点。
    #   若是二维目标，就优先删掉 HV 贡献小的点，有利于提升最终 HV。
    #   只适用于 2 目标且默认是最小化问题。
    # -------------------------------------------------------------------------
    """
    Approximate 2D hypervolume contributions for a NON-DOMINATED set (minimization).
    For each point i (sorted by f1 asc), its unique contribution is:
        (x_{i+1}-x_i) * (y_{i-1}-y_i)
    with y_0 = ref_y, x_{n+1} = ref_x.

    Returns contrib in the ORIGINAL order of F.
    """
    F = np.asarray(F, dtype=float)
    if F.shape[0] == 0:
        return np.array([], dtype=float)

    idx = np.argsort(F[:, 0], kind="mergesort")
    x = F[idx, 0]
    y = F[idx, 1]

    x_next = np.concatenate([x[1:], [ref_point[0]]])
    y_prev = np.concatenate([[ref_point[1]], y[:-1]])

    contrib = (x_next - x) * (y_prev - y)
    contrib = np.maximum(contrib, 0.0)

    out = np.empty_like(contrib)
    out[idx] = contrib
    return out


def _truncate_archive_hv2d(
    X: np.ndarray,
    F: np.ndarray,
    max_size: int,
    ref_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # -------------------------------------------------------------------------
    # 功能：
    #   对二维非支配 archive 做基于 HV 贡献的截断。
    #
    #   1) 保护两端极值点，避免边界覆盖被破坏
    #   2) 分批删除 HV 贡献最小的点，兼顾效果和速度
    #
    #   多目标优化不仅要“接近前沿”，还要“覆盖均匀”。
    #   archive 超限时如果随便删点，可能会导致前沿两端塌陷或中间断裂。
    # -------------------------------------------------------------------------
    """
    Truncate a NON-DOMINATED 2D archive by (approx.) hypervolume contribution.

    Improvements vs. naive pruning:
    - Protect extreme points (best f1 and best f2) to preserve coverage.
    - Remove points in small batches for speed.

    Notes:
    - Works for minimization problems.
    - For >2 objectives fall back to crowding (handled outside this function).
    """
    if F.shape[0] <= max_size:
        return X, F

    F = np.asarray(F, dtype=float)
    X = np.asarray(X, dtype=float)

    if ref_point is None:
        fmax = F.max(axis=0)
        fmin = F.min(axis=0)
        margin = 0.1 * (np.abs(fmax - fmin) + 1e-12)
        ref_point = fmax + margin

    n = F.shape[0]
    keep = np.ones(n, dtype=bool)

    # protect extremes (min of each objective)
    protected = set()
    protected.add(int(np.argmin(F[:, 0])))
    protected.add(int(np.argmin(F[:, 1])))

    # prune in batches to reduce recomputation overhead
    while keep.sum() > max_size:
        idx = np.where(keep)[0]
        contrib = _hv2d_contributions(F[idx], ref_point)

        # map protected -> local indices and avoid removing them
        if protected:
            prot_local = [np.where(idx == p)[0][0] for p in protected if p in idx]
            if len(prot_local) > 0:
                contrib = contrib.copy()
                contrib[prot_local] = np.inf

        # remove up to 5% per round (at least 1)
        rm_cnt = int(max(1, min(keep.sum() - max_size, max(1, 0.05 * keep.sum()))))
        rm_local = np.argsort(contrib)[:rm_cnt]
        keep[idx[rm_local]] = False

    return X[keep], F[keep]


def update_archive(
    arch_X: Optional[np.ndarray],
    arch_F: Optional[np.ndarray],
    pop_X: np.ndarray,
    pop_F: np.ndarray,
    max_size: Optional[int] = None,
    truncation: str = "hv2d",
) -> Tuple[np.ndarray, np.ndarray]:
    # -------------------------------------------------------------------------
    # 功能：
    #   把旧 archive和当前种群合并，然后只保留非支配解。
    #   如果数量超过上限，再按照指定策略截断。
    # 流程：
    #   1) 合并 archive + population
    #   2) 提取非支配前沿
    #   3) 若超上限：
    #      - 2目标时可用 hv2d 截断
    #      - 其他情况通常退回 crowding 截断
    # -------------------------------------------------------------------------
    """
    MATLAB UpdateArchive.m equivalent:
    Merge archive + population, keep non-dominated. Optionally truncate.

    truncation:
      - "hv2d": (only if n_obj==2) truncate by 2D HV contribution (often improves HV)
      - "crowding": truncate by crowding distance (NSGA-II style)
    """
    if arch_X is None or arch_X.size == 0:
        X = pop_X
        F = pop_F
    else:
        X = np.vstack([arch_X, pop_X])
        F = np.vstack([arch_F, pop_F])

    nds = NonDominatedSorting()
    nd_idx = nds.do(F, only_non_dominated_front=True)
    X_nd, F_nd = X[nd_idx], F[nd_idx]

    if max_size is not None and X_nd.shape[0] > max_size:
        if truncation.lower() == "hv2d" and F_nd.shape[1] == 2:
            X_nd, F_nd = _truncate_archive_hv2d(X_nd, F_nd, max_size=max_size)
        else:
            cd = calc_crowding_distance(F_nd)
            order = np.argsort(-cd)  # descending
            X_nd = X_nd[order[:max_size]]
            F_nd = F_nd[order[:max_size]]

    return X_nd, F_nd


# =============================================================================
# Leader selection
# =============================================================================
# 这一部分负责“从 archive 里选谁来引导当前个体”。
# 一部分用 decomposition/Tchebycheff 方式选，强调覆盖
# 一部分用 crowding 概率抽样选，强调多样性


def _select_leaders_crowding(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    n_leaders: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   按拥挤距离的概率分布，从 archive 中抽样 leader。
    #
    #   拥挤距离大的点附近更稀疏，更能代表“尚未充分覆盖”的区域，
    #   因而应更常被当作 leader。
    #
    #   提高 Pareto 前沿分布的均匀性与多样性。
    # -------------------------------------------------------------------------
    """
    Sample leaders from the archive proportional to crowding distance.
    """
    n_arch = arch_X.shape[0]
    if n_arch == 0:
        raise ValueError("Archive empty - cannot select leaders.")

    cd = calc_crowding_distance(arch_F)
    if np.any(np.isinf(cd)):
        finite = cd[np.isfinite(cd)]
        max_finite = finite.max() if finite.size > 0 else 1.0
        cd = cd.copy()
        cd[np.isinf(cd)] = 2.0 * max_finite

    cd = np.nan_to_num(cd, nan=0.0, posinf=1e9, neginf=0.0)
    if cd.sum() <= 0:
        idx = rng.integers(0, n_arch, size=n_leaders)
    else:
        p = cd / cd.sum()
        idx = rng.choice(n_arch, size=n_leaders, replace=True, p=p)

    return arch_X[idx]


def _make_weights(pop_size: int, n_obj: int, rng: np.random.Generator) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   为分解式 leader 选择生成权向量。
    #   2目标：用一条线性网格，便于均匀覆盖前沿方向
    #   多目标：用 Dirichlet 随机生成，快速且实用
    #   定义了“不同搜索方向”。
    # -------------------------------------------------------------------------
    """
    Weight vectors for decomposition-based leader selection.
    """
    if n_obj == 2:
        w1 = np.linspace(0.01, 0.99, pop_size)
        W = np.column_stack([w1, 1.0 - w1])
        return W
    # for many-objective: random Dirichlet (good enough, fast)
    W = rng.dirichlet(np.ones(n_obj), size=pop_size)
    return W


def _select_leaders_tchebycheff(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   基于加权 Tchebycheff 标量化，从 archive 中为每个权向量挑选 leader。

    #   不同的权向量代表 Pareto 前沿上的不同偏好方向。
    #   对每个方向，都从 archive 里选一个“最适合该方向”的代表解。

    #   强化前沿覆盖，尤其对凹前沿通常比简单加权和更稳。
    # -------------------------------------------------------------------------
    """
    Decomposition-based leader selection using weighted Tchebycheff:
        g(x|w) = max_j w_j * |f_j - z_j|
    where z is the ideal point of the archive.

    For concave Pareto fronts this is typically better than plain weighted sum.
    """
    z = arch_F.min(axis=0)
    diff = np.abs(arch_F - z)  # (A, m)

    # Compute g for each weight vector (N, A)
    # g[i, a] = max_j W[i,j] * diff[a,j]
    g = np.max(diff[None, :, :] * W[:, None, :], axis=2)
    idx = np.argmin(g, axis=1)  # (N,)

    return arch_X[idx]


def select_leaders_hybrid(
    arch_X: np.ndarray,
    arch_F: np.ndarray,
    n_leaders: int,
    rng: np.random.Generator,
    p_decomp: float = 0.7,
    W_base: Optional[np.ndarray] = None,
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 功能：
    #   混合式多领导者选择。
    #
    #   对每一个 leader 名额：
    #   - 以 p_decomp 的概率，使用分解式/Tchebycheff 选 leader（偏覆盖）
    #   - 否则使用 crowding 概率抽样选 leader（偏多样性）

    # -------------------------------------------------------------------------

    n_arch = arch_X.shape[0]
    if n_arch == 0:
        raise ValueError("Archive empty - cannot select leaders.")

    n_obj = arch_F.shape[1]

    use_decomp = rng.random(n_leaders) < p_decomp
    leaders = np.empty((n_leaders, arch_X.shape[1]), dtype=float)

    # crowding leaders
    if np.any(~use_decomp):
        leaders[~use_decomp] = _select_leaders_crowding(
            arch_X, arch_F, int((~use_decomp).sum()), rng
        )

    # decomposition leaders
    if np.any(use_decomp):
        if W_base is None:
            W = rng.dirichlet(np.ones(n_obj), size=int(use_decomp.sum()))
        else:
            # shuffle to cover PF uniformly (prefer no replacement if possible)
            n_w = int(use_decomp.sum())
            if n_w <= W_base.shape[0]:
                W = W_base[rng.permutation(W_base.shape[0])[:n_w]]
            else:
                W = W_base[rng.integers(0, W_base.shape[0], size=n_w)]
        leaders[use_decomp] = _select_leaders_tchebycheff(arch_X, arch_F, W)

    return leaders


# =============================================================================
# Main optimizer
# =============================================================================
#   1) 初始化
#   2) archive 建立
#   3) 每代 rank/crowding 计算
#   4) 三层个体更新（m1/m2/m3）
#   5) SBX + 变异
#   6) NSGA-II风格生存选择
#   7) archive 更新


@dataclass
class MOIVYResult:
    # 用 dataclass 封装最终输出，方便调用者直接访问：
    # - X, F：最终种群
    # - archive_X, archive_F：外部存档中的非支配解
    # - history：每代日志（当前主要记录 archive_size）
    X: np.ndarray
    F: np.ndarray
    archive_X: np.ndarray
    archive_F: np.ndarray
    history: List[Dict[str, Any]]


def moivy_optimize(
    problem,
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 1,
    archive_size: Optional[int] = 200,
    # --- leader & neighborhood ---
    neighbor_random_rate: float = 0.35,
    leader_p_decomp: float = 0.75,
    # --- variation operators ---
    use_sbx: bool = True,
    sbx_eta: float = 15.0,
    p_sbx_start: float = 0.10,
    p_sbx_end: float = 0.40,
    mutation_prob: Optional[float] = None,
    mutation_eta_start: float = 12.0,
    mutation_eta_end: float = 40.0,
    # --- archive ---
    archive_truncation: str = "hv2d",
    # --- bounds & init ---
    boundary_handling: str = "reflect",
    init_method: str = "lhs",
    # --- misc ---
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose: bool = False,
) -> MOIVYResult:

    # 初始化随机数发生器，保证实验可复现
    rng = np.random.default_rng(seed)

    n_var = int(problem.n_var)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    if xl.ndim == 0:
        xl = np.full(n_var, float(xl))
    if xu.ndim == 0:
        xu = np.full(n_var, float(xu))
    var_range = np.where(xu - xl == 0, 1.0, xu - xl)

    # 预先生成一组权向量，供分解式 leader 选择使用。
    # 这样每代就不需要完全重新构造，效率更高，也能保持一定的方向覆盖稳定性。
    # Precompute weight vectors for decomposition leaders
    W_base = _make_weights(pop_size=pop_size, n_obj=int(problem.n_obj), rng=rng)

    # ---- initialization ----
    # 初始化种群位置 X。
    # 默认采用 LHS，使初始样本分布更均匀，减少早期搜索盲区。
    if init_method.lower() == "lhs":
        X = lhs_sample(rng, xl, xu, pop_size)
    else:
        X = rng.uniform(xl, xu, size=(pop_size, n_var))

    # 计算初始种群的目标函数值
    F = problem.evaluate(X, return_values_of=["F"])

    # GV = growth vector（生长向量）
    # 这里不是严格物理意义上的速度，而是一个“带记忆的生长趋势”。
    # 相比每代重新归一化位置，保留 GV 记忆可以让搜索更平滑。
    # GV in [-0.5, 0.5]
    GV = (X - xl) / var_range - 0.5

    # 建立初始外部存档：
    # 把初始种群中的非支配解先存起来，后面 leader 就主要从这里选。
    archive_X, archive_F = update_archive(
        None, None, X, F, max_size=archive_size, truncation=archive_truncation
    )

    history: List[Dict[str, Any]] = []

    # =========================
    # 主迭代开始
    # =========================
    for gen in range(1, n_gen + 1):

        # 先对当前种群做非支配排序与拥挤距离计算
        # rank 决定层级身份，crowd 主要用于保留阶段
        rank, crowd, _ = fast_non_dominated_sort_and_crowding(F)

        # neighbors: cyclic + random mixing
        # 邻居机制：
        # 默认使用循环邻居（第 i 个看第 i+1 个），同时以一定概率换成随机邻居。
        # 这样兼顾了稳定局部结构和随机扰动。
        idx = np.arange(pop_size)
        neigh = np.roll(idx, -1)
        if neighbor_random_rate > 0:
            rand_neigh = rng.integers(0, pop_size, size=pop_size)
            rand_neigh[rand_neigh == idx] = (rand_neigh[rand_neigh == idx] + 1) % pop_size
            mask_rn = rng.random(pop_size) < neighbor_random_rate
            neigh[mask_rn] = rand_neigh[mask_rn]
        Xn = X[neigh]

        # leaders from archive
        # 当前代的领导者优先从 archive 中选，而不是只看当前种群。
        # 向“历史累计的优秀非支配解”学习。
        if archive_X is None or archive_X.shape[0] == 0:
            leaders = X[rng.integers(0, pop_size, size=pop_size)]
        else:
            leaders = select_leaders_hybrid(
                archive_X, archive_F, pop_size, rng, p_decomp=leader_p_decomp, W_base=W_base
            )

        # differential exploration vectors
        # 差分探索向量：X[r1] - X[r2]
        # 这一项为个体提供更强的跨区域跳跃能力，尤其对被支配个体很重要。
        r1 = rng.integers(0, pop_size, size=pop_size)
        r2 = rng.integers(0, pop_size, size=pop_size)
        eq = r1 == r2
        r2[eq] = (r2[eq] + 1) % pop_size
        diff = X[r1] - X[r2]

        # adaptive coefficient (exploration -> exploitation)
        # a 从 1 逐步减小到接近 0：
        # 前期更强调探索，后期更强调收敛与开发。
        a = 1.0 - gen / n_gen

        beta1 = 1.0 + 0.5 * rng.random((pop_size, 1))
        absN = np.abs(rng.standard_normal(size=(pop_size, n_var)))
        N = rng.standard_normal(size=(pop_size, n_var))
        r = rng.random((pop_size, 1))

        Y = np.empty_like(X)

        # 三层分层：
        # m1: 精英层（当前种群中的第一非支配前沿）
        # m2: 次精英层（第二前沿）
        # m3: 其余被支配层
        m1 = rank == 0
        m2 = rank == 1
        m3 = ~(m1 | m2)

        # ---- ivy offspring generation (vectorized) ----
        # - 精英层：偏局部开发与细化
        # - 次精英层：开发与探索平衡
        # - 被支配层：偏全局探索与跳出停滞
        if np.any(m1):
            # elite: local refinement + mild leader attraction
            # 精英层：
            # - 主要围绕邻域做细化
            # - 只保留较弱 leader 牵引，避免把优质解拉坏
            # - 适度保留 GV 随机生长，维持柔性
            Y[m1] = (
                X[m1]
                + (0.9 * beta1[m1]) * absN[m1] * (Xn[m1] - X[m1])
                + (0.15 * a) * (leaders[m1] - X[m1])
                + (0.15 + 0.25 * a) * N[m1] * GV[m1]
            )

        if np.any(m2):
            # near-elite: leader guidance + neighbor + diff exploration
            # 次精英层：
            # - 既向 leader 学习
            # - 又结合邻域项和差分项
            # - 是整个人群中最“均衡”的那一层
            Y[m2] = (
                X[m2]
                + (0.9 * a + 0.05) * (leaders[m2] - X[m2])
                + 0.35 * beta1[m2] * absN[m2] * (Xn[m2] - X[m2])
                + (0.25 + 0.25 * a) * diff[m2]
                + (0.15 + 0.25 * a) * N[m2] * GV[m2]
            )

        if np.any(m3):
            # dominated: global exploration (diff) + leader guidance
            # 被支配层：
            # - 更强的差分探索项
            # - 更明显的 leader 引导
            # - 目标是尽快摆脱劣势区域，寻找新的前沿片段
            Y[m3] = (
                X[m3]
                + (0.55 + 0.45 * a) * diff[m3]
                + (0.7 * a + 0.15) * r[m3] * (leaders[m3] - X[m3])
                + (0.20 + 0.25 * a) * N[m3] * GV[m3]
            )

        # ---- optional SBX "ivy grafting" (same budget, replaces some ivy offspring) ----
        # 这里把部分 Ivy 产生的后代替换成 SBX 重组后的后代。
        # 可理解为：当前个体与 leader 做一次“嫁接”，快速形成新的潜在优解。
        if use_sbx and sbx_eta is not None and sbx_eta > 0:
            p_sbx = p_sbx_start + (p_sbx_end - p_sbx_start) * (gen / n_gen)
            mask = rng.random(pop_size) < p_sbx
            if np.any(mask):
                Y_sbx = sbx_one_child(X[mask], leaders[mask], xl, xu, eta=sbx_eta, rng=rng)
                Y[mask] = Y_sbx

        # boundary handling
        # 处理越界问题，默认使用 reflect 反射。
        if boundary_handling.lower() == "reflect":
            Y = reflect_bounds(Y, xl, xu)
        else:
            np.clip(Y, xl, xu, out=Y)

        # adaptive polynomial mutation
        # 再做一层自适应多项式变异。
        # 早期 eta 较小、步长偏大；后期 eta 较大、步长更细。
        eta = mutation_eta_start + (mutation_eta_end - mutation_eta_start) * (gen / n_gen)
        if eta is not None and eta > 0:
            Y = polynomial_mutation(Y, xl, xu, eta=float(eta), prob=mutation_prob, rng=rng)

        # evaluate offspring (vectorized)
        # 计算新生成后代的目标值
        FY = problem.evaluate(Y, return_values_of=["F"])

        # GV update: keep memory + new movement direction (velocity-like)
        # 新的 GV 不是直接替换旧 GV，而是“旧记忆 + 新位移方向”的加权融合。
        # 这就是 growth vector 的“记忆性”体现。
        GVY = 0.7 * GV + 0.3 * ((Y - X) / var_range)

        # ---- survival (NSGA-II style rank + crowding) ----
        # 生存选择：
        # 把父代和子代合并后，再按 rank 优先、crowding 次优的方式筛回 pop_size 个。
        # 这是典型的 NSGA-II 风格保留策略。
        Xc = np.vstack([X, Y])
        Fc = np.vstack([F, FY])
        GVc = np.vstack([GV, GVY])

        rank_c, crowd_c, _ = fast_non_dominated_sort_and_crowding(Fc)
        order = np.lexsort((-crowd_c, rank_c))  # rank asc, crowd desc
        sel = order[:pop_size]

        X, F, GV = Xc[sel], Fc[sel], GVc[sel]

        # ---- external archive update ----
        # 每代生存选择完成后，再把当前种群并入 archive。
        # 这样 archive 能跨代持续维护更完整的非支配解集。
        archive_X, archive_F = update_archive(
            archive_X, archive_F, X, F, max_size=archive_size, truncation=archive_truncation
        )

        # ---- logging ----
        # 记录当前代信息，便于后续画图或分析收敛过程。
        history.append({"gen": gen, "archive_size": int(archive_F.shape[0])})

        if callback is not None:
            callback(
                {
                    "gen": gen,
                    "X": X,
                    "F": F,
                    "archive_X": archive_X,
                    "archive_F": archive_F,
                    "rank": rank,
                    "crowd": crowd,
                }
            )

        if verbose and (gen == 1 or gen % max(1, n_gen // 10) == 0):
            print(f"[HMOIVY] Gen {gen}/{n_gen} | Archive size: {archive_F.shape[0]}")

    return MOIVYResult(X=X, F=F, archive_X=archive_X, archive_F=archive_F, history=history)

