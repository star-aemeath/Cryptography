import numpy as np
import math
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce
import random
# 解决中文乱码
plt.rcParams["font.family"] = ["SimHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False

# 两个数最小公倍数
def lcm(a, b):
    return a * b // gcd(a, b)

# 列表所有数的最小公倍数（求置换总阶核心）
def lcm_list(arr):
    return reduce(lcm, arr)

def logistic_seq(mu, x0, M, N):
    x = x0
    # 第一步：预迭代M轮，丢弃暂态值（题目要求M常取1000）
    for _ in range(M):
        x = mu * x * (1 - x)
    seq = []
    # 第二步：迭代N轮，生成用于置乱的N个混沌值
    for _ in range(N):
        x = mu * x * (1 - x)
        seq.append(x)
    return np.array(seq)

def tent_seq(r, x0, M, N):
    x = x0
    # 预迭代消暂态
    for _ in range(M):
        if x < 0.5:
            x = r * x
        else:
            x = r * (1 - x)
    seq = []
    for _ in range(N):
        if x < 0.5:
            x = r * x
        else:
            x = r * (1 - x)
        seq.append(x)
    return np.array(seq)

def henon_seq(a, b, x0, y0, M, N):
    x, y = x0, y0
    # 预迭代M轮消暂态
    for _ in range(M):
        x_next = 1 - a * x**2 + y
        y_next = b * x
        x, y = x_next, y_next
    seq = []
    for _ in range(N):
        x_next = 1 - a * x**2 + y
        y_next = b * x
        x, y = x_next, y_next
        seq.append(x)  # 取x分量作为混沌序列
    return np.array(seq)

def trim_mean(arr, trim_rate=0.05):
    arr_sorted = sorted(arr)
    cut = int(len(arr_sorted)*trim_rate)
    return np.mean(arr_sorted[cut:-cut])

def build_permutation(chaos_seq):
    N = len(chaos_seq)
    # argsort：返回「从小到大排序后，原元素的下标列表」
    perm = np.argsort(chaos_seq)
    return perm

def decompose_cycles(perm):
    N = len(perm)
    visited = [False]*N  # 标记每个下标是否已遍历
    cycles = []          # 存储每一个循环圈的长度
    for i in range(N):
        if not visited[i]:
            cur = i
            cycle_len = 0
            # 沿着置换走，直到回到起点，形成一个循环
            while not visited[cur]:
                visited[cur] = True
                cur = perm[cur]
                cycle_len +=1
            cycles.append(cycle_len)
    # 统计：{循环长度: 该长度循环圈的个数}
    len_count = {}
    for l in cycles:
        len_count[l] = len_count.get(l,0)+1
    # 计算置换总阶：所有循环长度的最小公倍数
    total_order = lcm_list(cycles)
    return len_count, total_order, cycles

def single_test(mapping_type, N, seed, M=1000):
    # 根据选择的映射调用对应序列生成函数
    if mapping_type == "logistic":
        mu = 3.8
        seq = logistic_seq(mu, seed, M, N)
    elif mapping_type == "tent":
        r = 1.99
        seq = tent_seq(r, seed, M, N)
    elif mapping_type == "henon":
        a, b = 1.4, 0.3
        y0 = seed + 0.1  # 用种子偏移生成y初始值
        seq = henon_seq(a, b, seed, y0, M, N)
    perm = build_permutation(seq)
    len_cnt, order, _ = decompose_cycles(perm)
    return len_cnt, order

def batch_avg_order(mapping, N_list, test_seeds):
    avg_orders = []
    for N in N_list:
        orders = []
        # 多个种子分别测试，取均值消除随机波动
        for s in test_seeds:
            _, o = single_test(mapping, N, s)
            orders.append(o)
        avg_orders.append(trim_mean(orders))
        #avg_orders.append(np.mean(orders))
    return avg_orders

if __name__ == "__main__":
    # 实验参数配置
    #N_range=[10,20,30,40,50,60,70,80,90,100]
    N_range = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]  # 不同置乱长度
    seed_num = 2000  # 自定义要生成多少个随机种子
    test_seeds = np.random.uniform(low=1e-7, high=1.0 - 1e-7, size=seed_num).tolist()
    case_seed=test_seeds[0]
    print("本次随机生成的种子列表：", test_seeds)

    maps = ["logistic","tent","henon"]
    labels = ["Logistic","Tent","Henon"]

    # 单组详细案例打印输出（Logistic N=50）
    print(f"===== 同一随机种子{case_seed:.6f}，N=50 三种映射循环对照 =====")
    
    # Logistic
    dist_log, ord_log = single_test("logistic", N=50, seed=case_seed)
    print("\n【Logistic映射】")
    print(f"循环长度分布 {{长度:数量}}: {dist_log}")
    print(f"置换总阶(所有循环LCM): {ord_log}")

    # Tent
    dist_tent, ord_tent = single_test("tent", N=50, seed=case_seed)
    print("\n【Tent映射】")
    print(f"循环长度分布 {{长度:数量}}: {dist_tent}")
    print(f"置换总阶(所有循环LCM): {ord_tent}")

    # Henon
    dist_hen, ord_hen = single_test("henon", N=50, seed=case_seed)
    print("\n【Henon映射】")
    print(f"循环长度分布 {{长度:数量}}: {dist_hen}")
    print(f"置换总阶(所有循环LCM): {ord_hen}")
    
    # 批量计算每种映射在各N下的平均阶
    curve_data = []
    for m in maps:
        avg = batch_avg_order(m, N_range, test_seeds)
        curve_data.append(avg)

    # 绘制「平均阶-N」对比曲线
    plt.figure(figsize=(10,6))
    for i, data in enumerate(curve_data):
        plt.plot(N_range, data, marker='o', linewidth=2, label=labels[i])
    plt.xlabel("置乱长度 N")
    plt.ylabel("置换平均阶")
    plt.title("三种混沌映射 平均置换阶-N关系曲线")
    plt.legend()
    # 开启对数纵轴+完整网格
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.show()


    
