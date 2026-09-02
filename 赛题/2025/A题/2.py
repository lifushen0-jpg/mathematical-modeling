import numpy as np
from scipy.optimize import differential_evolution
# ==================== 1. 常数 ====================
g = 9.8
R_SMOKE = 10.0
T_SMOKE = 20.0
V_SINK = 3.0
# M1
M0 = np.array([20000.0, 0.0, 2000.0])
v_M = -300 * M0 / np.linalg.norm(M0)
T_HIT = np.linalg.norm(M0) / 300
# FY1
FY1 = np.array([17800.0, 0.0, 1800.0])
MAX_DELAY = np.sqrt(2 * FY1[2] / g)
# ==================== 2. 圆柱目标采样 ====================
def make_target_points(n_theta=60, n_z=3, n_r=2):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    points = []
    # 圆柱侧面
    for z in np.linspace(0, 10, n_z):
        points.append(np.column_stack((
            7 * np.cos(theta),
            200 + 7 * np.sin(theta),
            np.full(n_theta, z)
        )))
    # 上下底面圆心
    points.append(np.array([
        [0.0, 200.0, 0.0],
        [0.0, 200.0, 10.0]
    ]))
    # 上下底面同心圆采样
    for z in [0.0, 10.0]:
        for r in np.linspace(7 / n_r, 7, n_r):
            points.append(np.column_stack((
                r * np.cos(theta),
                200 + r * np.sin(theta),
                np.full(n_theta, z)
            )))
    return np.vstack(points)
# DE 搜索使用粗网格
TARGET_COARSE = make_target_points(
    n_theta=60,
    n_z=3,
    n_r=2
)
# 最终验证使用细网格
TARGET_FINE = make_target_points(
    n_theta=180,
    n_z=7,
    n_r=4
)
# ==================== 3. 烟幕中心到所有视线段的最大距离 ====================
def max_sight_distance(missiles, smokes, target_points, chunk_size=128):
    T = len(missiles)
    result = np.empty(T)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        M = missiles[start:end]
        C = smokes[start:end]
        # M -> P
        MP = target_points[None, :, :] - M[:, None, :]
        # M -> C
        MC = C - M
        # 内积
        dot = np.einsum(
            "tnk,tk->tn",
            MP,
            MC
        )
        MP2 = np.einsum(
            "tnk,tnk->tn",
            MP,
            MP
        )
        # 垂足在线段上的投影参数
        lam = dot / MP2
        lam = np.clip(lam, 0, 1)
        MC2 = np.einsum(
            "tk,tk->t",
            MC,
            MC
        )[:, None]
        # |MC-lambda*MP|^2
        d2 = (
            MC2
            - 2 * lam * dot
            + lam**2 * MP2
        )
        d2 = np.maximum(d2, 0)
        # 所有圆柱采样点中最难遮蔽的点
        result[start:end] = np.sqrt(
            np.max(d2, axis=1)
        )
    return result
# ==================== 4. 插值计算遮蔽时长 ====================
def coverage_duration(times, distances):
    """
    distances = D(t)
    D(t) <= 10 表示完全遮蔽。
    在线性插值下计算 D(t)=10 的交点，
    避免最终遮蔽时间只能是 dt 的整数倍。
    """
    f = distances - R_SMOKE
    total = 0.0
    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        f0 = f[i]
        f1 = f[i + 1]
        dt = t1 - t0
        # 两端均有效
        if f0 <= 0 and f1 <= 0:
            total += dt
        # 有效 -> 无效
        elif f0 <= 0 < f1:
            alpha = -f0 / (f1 - f0)
            total += alpha * dt
        # 无效 -> 有效
        elif f0 > 0 >= f1:
            alpha = f0 / (f0 - f1)
            total += (1 - alpha) * dt
    return total
# ==================== 5. 评价一组投放策略 ====================
def evaluate_strategy(x, target_points, dt=0.05, details=False):
    """
    x = [theta, speed, t_drop, delay]
    theta  : 航向角
    speed  : FY1速度
    t_drop : 投弹时刻
    delay  : 投弹后起爆延迟
    """
    theta, speed, t_drop, delay = x
    # ---------- 约束 ----------
    if not (0 <= theta <= 2 * np.pi):
        return None if details else -1e10
    if not (70 <= speed <= 140):
        return None if details else -1e10
    if t_drop < 0 or delay < 0:
        return None if details else -1e10
    if delay > MAX_DELAY:
        return None if details else -1e10
    t_explode = t_drop + delay
    if t_explode >= T_HIT:
        return None if details else -1e10
    # ---------- 无人机速度 ----------
    direction = np.array([
        np.cos(theta),
        np.sin(theta),
        0.0
    ])
    velocity = speed * direction
    # ---------- 投弹点 ----------
    P_drop = FY1 + velocity * t_drop
    # ---------- 起爆点 ----------
    P_explode = P_drop + velocity * delay
    P_explode = P_explode.copy()
    P_explode[2] -= 0.5 * g * delay**2
    if P_explode[2] < 0:
        return None if details else -1e10
    # ---------- 烟幕有效时间 ----------
    t_end = min(
        t_explode + T_SMOKE,
        T_HIT
    )
    times = np.arange(
        t_explode,
        t_end,
        dt
    )
    if len(times) == 0 or times[-1] < t_end:
        times = np.append(times, t_end)
    # ---------- 导弹位置 ----------
    missiles = (
        M0
        + times[:, None] * v_M
    )
    # ---------- 烟幕中心 ----------
    smokes = np.repeat(
        P_explode[None, :],
        len(times),
        axis=0
    )
    smokes[:, 2] -= (
        V_SINK
        * (times - t_explode)
    )
    # ---------- 最大视线距离 ----------
    D = max_sight_distance(
        missiles,
        smokes,
        target_points
    )
    # ---------- 有效遮蔽时间 ----------
    T_cover = coverage_duration(
        times,
        D
    )
    # 最接近有效遮蔽状态时的距离
    min_D = np.min(D)
    if details:
        return {
            "cover_time": T_cover,
            "theta": theta,
            "speed": speed,
            "t_drop": t_drop,
            "delay": delay,
            "t_explode": t_explode,
            "P_drop": P_drop,
            "P_explode": P_explode,
            "times": times,
            "D": D,
            "min_D": min_D
        }
    # DE 默认做最小化
    # 第一项：最大化遮蔽时间
    # 第二项：极小权重辅助 DE 处理大量 T=0 的区域
    return -T_cover + 1e-6 * min_D
# ==================== 6. 差分进化目标函数 ====================
def objective(x):
    return evaluate_strategy(
        x,
        target_points=TARGET_COARSE,
        dt=0.05
    )
# ==================== 7. DE搜索 ====================
bounds = [
    (0, 2 * np.pi),   # theta
    (70, 140),        # speed
    (0, T_HIT),       # t_drop
    (0, MAX_DELAY)    # delay
]
def run_de(seed):
    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        popsize=15,
        maxiter=150,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-7,
        seed=seed,
        polish=False,
        updating="immediate",
        workers=1
    )
    return result
# ==================== 8. 多次独立运行 ====================
seeds = [1, 7, 19, 42, 123]
best_result = None
best_cover = -np.inf
for seed in seeds:
    result = run_de(seed)
    detail = evaluate_strategy(
        result.x,
        target_points=TARGET_COARSE,
        dt=0.02,
        details=True
    )
    print(
        f"seed={seed}, "
        f"T={detail['cover_time']:.6f}s"
    )
    if detail["cover_time"] > best_cover:
        best_cover = detail["cover_time"]
        best_result = result
# ==================== 9. 高精度复算 ====================
x_best = best_result.x
final = evaluate_strategy(
    x_best,
    target_points=TARGET_FINE,
    dt=0.005,
    details=True
)
theta = final["theta"]
speed = final["speed"]
t_drop = final["t_drop"]
delay = final["delay"]
# ==================== 10. 输出 ====================
print("\n========== 最优方案 ==========")
print(f"航向角 = {theta:.8f} rad")
print(f"航向角 = {np.degrees(theta):.6f}°")
print(f"无人机速度 = {speed:.6f} m/s")
print(f"投弹时刻 = {t_drop:.6f} s")
print(f"起爆延迟 = {delay:.6f} s")
print(f"起爆时刻 = {final['t_explode']:.6f} s")
print("投弹点 =", final["P_drop"])
print("起爆点 =", final["P_explode"])
print(f"有效遮蔽时间 = {final['cover_time']:.6f} s")
# ==================== 11. 寻找遮蔽区间 ====================
times = final["times"]
D = final["D"]
f = D - 10
intervals = []
start = None
if f[0] <= 0:
    start = times[0]
for i in range(len(times) - 1):
    if f[i] > 0 and f[i + 1] <= 0:
        alpha = f[i] / (f[i] - f[i + 1])
        start = times[i] + alpha * (times[i + 1] - times[i])
    elif f[i] <= 0 and f[i + 1] > 0:
        alpha = -f[i] / (f[i + 1] - f[i])
        end = times[i] + alpha * (times[i + 1] - times[i])
        intervals.append((start, end))
        start = None
if start is not None:
    intervals.append((start, times[-1]))
print("有效遮蔽区间：")
for a, b in intervals:
    print(
        f"[{a:.6f}, {b:.6f}] s, "
        f"长度 = {b-a:.6f} s"
    )