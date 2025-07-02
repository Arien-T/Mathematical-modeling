import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


data = {
    '零配件1次品率': [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],
    '零配件1购买单价': [4, 4, 4, 4, 4, 4],
    '零配件1检测成本': [2, 2, 2, 1, 8, 2],
    '零配件2次品率': [0.10, 0.20, 0.10, 0.20, 0.20, 0.05],
    '零配件2购买单价': [18, 18, 18, 18, 18, 18],
    '零配件2检测成本': [3, 3, 3, 1, 1, 3],
    '成品次品率': [0.10, 0.20, 0.10, 0.20, 0.10, 0.05],
    '成品装配成本': [6, 6, 6, 6, 6, 6],
    '成品检测成本': [3, 3, 3, 2, 2, 3],
    '市场售价': [56, 56, 56, 56, 56, 56],
    '调换损失': [6, 6, 30, 30, 10, 10],
    '拆解费用': [5, 5, 5, 5, 5, 40]
}


df = pd.DataFrame(data)


xunhuancishu = 10  # 最多售卖次数
yuzhi = 0.01  # 次品率阈值
dijainbili1 = 0.8  # 零配件1次品率递减比例
dijainbili2 = 0.8  # 零配件2次品率递减比例
dijainbili3 = 0.9  # 成品次品率递减比例
yunxuwucha = 0.02  # 允许的误差

# 置信度Z值
zhixinzhi = {
    95: norm.ppf(0.975),
    90: norm.ppf(0.95)
}

# 抽样检测次品率估计函数
def cipinlvgujizhi(p_estimate, confidence, wucha1=yunxuwucha):
    zhixindu1 = zhixinzhi[confidence]
    yangbenshuliang = (zhixindu1**2 * p_estimate * (1 - p_estimate)) / (wucha1**2)
    yangbencipinlv = np.random.binomial(int(yangbenshuliang), p_estimate) / yangbenshuliang
    return yangbencipinlv

# 决策分析函数
def juece(row, confidence=95):
    zhixindu1 = zhixinzhi[confidence]
    

    cipinlvguji11 = cipinlvgujizhi(row['零配件1次品率'], confidence)
    cipinlvguji22 = cipinlvgujizhi(row['零配件2次品率'], confidence)
    cipinlvguji33 = cipinlvgujizhi(row['成品次品率'], confidence)


    costs = {
        '零配件1': (row['零配件1购买单价'], row['零配件1检测成本']),
        '零配件2': (row['零配件2购买单价'], row['零配件2检测成本']),
        '成品': (row['成品装配成本'], row['市场售价'], row['成品检测成本']),
        '拆解': (row['调换损失'], row['拆解费用'])
    }

    totalcost1, cycles = 0, 0
    cipinlishi = {'零配件1': [], '零配件2': [], '成品': []}

    while cycles < xunhuancishu and any(rate > yuzhi for rate in [cipinlvguji11, cipinlvguji22, cipinlvguji33]):
        cycles += 1

        cipinlishi['零配件1'].append(cipinlvguji11)
        cipinlishi['零配件2'].append(cipinlvguji22)
        cipinlishi['成品'].append(cipinlvguji33)

        # 计算零配件与成品的检测或替换成本
        jaincetihuanchengben1 = min(costs['零配件1'][1], cipinlvguji11 * costs['零配件1'][0])
        jaincetihuanchengben2 = min(costs['零配件2'][1], cipinlvguji22 * costs['零配件2'][0])
        product_cost = min(costs['成品'][2], cipinlvguji33 * (costs['成品'][0] + costs['成品'][1]))

        shoumaihaufei = jaincetihuanchengben1 + jaincetihuanchengben2 + product_cost

        # 如果需要拆解，计算额外成本
        if costs['拆解'][1] < costs['拆解'][0]:
            shoumaihaufei += costs['拆解'][1] + cipinlvguji11 * costs['零配件1'][0] + cipinlvguji22 * costs['零配件2'][0] + costs['成品'][0]

        totalcost1 += shoumaihaufei

        # 次品率递减
        cipinlvguji11 *= (1 - dijainbili1)
        cipinlvguji22 *= (1 - dijainbili2)
        cipinlvguji33 *= (1 - dijainbili3)

    return {
        '检测零配件1': costs['零配件1'][1] < cipinlvguji11 * costs['零配件1'][0],
        '检测零配件2': costs['零配件2'][1] < cipinlvguji22 * costs['零配件2'][0],
        '检测成品': costs['成品'][2] < cipinlvguji33 * (costs['成品'][0] + costs['成品'][1]),
        '拆解': costs['拆解'][1] < costs['拆解'][0],
        '总成本': totalcost1,
        '售卖次数': cycles,
        '零配件1次品率历史': cipinlishi['零配件1'],
        '零配件2次品率历史': cipinlishi['零配件2'],
        '成品次品率历史': cipinlishi['成品']
    }

# 对所有情况进行决策分析
zhixin95 = df.apply(juece, axis=1, confidence=95)
zhixin90 = df.apply(juece, axis=1, confidence=90)


biaozhun95 = pd.DataFrame(zhixin95.tolist())
biaozhun90 = pd.DataFrame(zhixin90.tolist())
biaozhun1 = pd.DataFrame(zhixin95.tolist(), columns=['检测零配件1', '检测零配件2', '检测成品', '拆解', '总成本', '售卖次数'])
biaozhun2 = pd.DataFrame(zhixin90.tolist(), columns=['检测零配件1', '检测零配件2', '检测成品', '拆解', '总成本', '售卖次数'])

print("\n" + "=" * 50)
print("95% 置信度下的决策分析结果：")
print("=" * 50)
print(biaozhun1[['检测零配件1', '检测零配件2', '检测成品', '拆解', '总成本', '售卖次数']])
print("=" * 50)

print("\n" + "=" * 50)
print("90% 置信度下的决策分析结果：")
print("=" * 50)
print(biaozhun2[['检测零配件1', '检测零配件2', '检测成品', '拆解', '总成本', '售卖次数']])
print("=" * 50)



def lishishujukeshihua(df_results, confidence_levels):
    num_plots = len(df_results)
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True, sharey=True)
    

    axes = axes.flatten()
    
    for i, (ax, (index, row)) in enumerate(zip(axes, df_results.iterrows())):
        cycles = range(1, len(row['零配件1次品率历史']) + 1)
        ax.plot(cycles, row['零配件1次品率历史'], label='零配件1次品率', marker='.', linestyle='--', color='red')
        ax.plot(cycles, row['零配件2次品率历史'], label='零配件2次品率', marker='.', linestyle='--', color='darkblue')
        ax.plot(cycles, row['成品次品率历史'], label='成品次品率', marker='.', color='orange')
        ax.set_title(f"情况 {index+1} - {confidence_levels[i]}")
        ax.set_xlabel('售卖次数')
        ax.set_ylabel('次品率')
        ax.grid(True)
        ax.legend()


    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


lishishujukeshihua(biaozhun95, ["95% 置信度"] * len(biaozhun95))
lishishujukeshihua(biaozhun90, ["90% 置信度"] * len(biaozhun90))