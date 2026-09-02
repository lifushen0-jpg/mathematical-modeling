import numpy as np

# ==============================
# 参数
# ==============================

g = 9.8
dt = 0.001

# M1 初始位置
M0 = np.array([20000.0, 0.0, 2000.0])

# M1 速度
v_M = -300 * M0 / np.linalg.norm(M0)

# FY1 初始位置
F0 = np.array([17800.0, 0.0, 1800.0])

# FY1 速度
v_F = np.array([-120.0, 0.0, 0.0])

# ==============================
# 投放点
# ==============================

t_drop = 1.5

P_drop = F0 + v_F * t_drop

# ==============================
# 起爆点
# ==============================

delay = 3.6
t_explode = t_drop + delay

P_explode = P_drop + v_F * delay
P_explode[2] -= 0.5 * g * delay**2

print("投放点:", P_drop)
print("起爆点:", P_explode)

# ==============================
# 真目标表面离散
# ==============================

points = []

# 圆柱侧面
theta = np.linspace(0, 2*np.pi, 360, endpoint=False)

for z in np.linspace(0, 10, 21):
    x = 7 * np.cos(theta)
    y = 200 + 7 * np.sin(theta)
    z_arr = np.full_like(theta, z)

    points.append(
        np.column_stack((x, y, z_arr))
    )

# 上、下底面
for z in [0, 10]:
    for r in np.linspace(0, 7, 15):

        x = r * np.cos(theta)
        y = 200 + r * np.sin(theta)
        z_arr = np.full_like(theta, z)

        points.append(
            np.column_stack((x, y, z_arr))
        )

target_points = np.vstack(points)

# ==============================
# 点到线段距离
# ==============================

def distance_to_segments(C, A, points):

    AB = points - A
    AC = C - A

    lam = (AB @ AC) / np.sum(AB**2, axis=1)

    # 限制在线段内
    lam = np.clip(lam, 0, 1)

    closest = A + lam[:, None] * AB

    distances = np.linalg.norm(
        closest - C,
        axis=1
    )

    return distances

# ==============================
# 时间扫描
# ==============================

times = np.arange(
    t_explode,
    t_explode + 20 + dt,
    dt
)

effective = []

for t in times:

    # M1 当前位置
    M = M0 + v_M * t

    # 烟幕中心
    C = P_explode.copy()
    C[2] -= 3 * (t - t_explode)

    # 烟幕中心到所有视线的距离
    distances = distance_to_segments(
        C,
        M,
        target_points
    )

    # 整个目标全部被遮挡
    if np.max(distances) <= 10:
        effective.append(t)

# ==============================
# 结果
# ==============================

if effective:

    print("开始遮蔽时刻:", effective[0])
    print("结束遮蔽时刻:", effective[-1])

    duration = len(effective) * dt

    print("有效遮蔽时间:", duration, "s")

else:

    print("没有有效遮蔽")
    