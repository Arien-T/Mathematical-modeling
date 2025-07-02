import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


data = {
    '零配件': ['零配件1', '零配件2', '零配件3', '零配件4', '零配件5', '零配件6', '零配件7', '零配件8'],
    '购买单价': [2, 8, 12, 2, 8, 12, 8, 12],
    '检测成本': [1, 1, 2, 1, 1, 2, 1, 2]
}
banchengpin = {
    '半成品': ['半成品1', '半成品2', '半成品3'],
    '装配成本': [8, 8, 8],
    '检测成本': [4, 4, 4],
    '拆解费用': [6, 6, 6]
}
chengpin = {
    '成品': ['成品'],
    '装配成本': [8],
    '检测成本': [6],
    '拆解费用': [10],
    '市场售价': [200],
    '调换损失': [40]
}


parts_df = pd.DataFrame(data)
semi_df = pd.DataFrame(banchengpin)
zuizhongjuedingshuchu = pd.DataFrame(chengpin)


shoumaicishu = 10  # 最大循环次数
yuzhi = 0.001  # 次品率阈值
dijianbili = 0.7  # 次品率递减比例
chouyangwucha = 0.02  # 抽样误差

# 估计次品率的抽样函数
def gujicipinlv(p_estimate, confidence_level):
    zhixinzhi = norm.ppf((1 + confidence_level) / 2)
    yangbenliang = (zhixinzhi**2 * p_estimate * (1 - p_estimate)) / chouyangwucha**2
    jaincezhi = np.random.binomial(int(yangbenliang), p_estimate) / yangbenliang
    return jaincezhi

# 定义总成本计算函数
def calculate_bufenchengben11(parts_df, semi_df, zuizhongjuedingshuchu, confidence_level=0.95, shoumaicishu=shoumaicishu, threshold=yuzhi):
    bufenchengben11 = 0
    cycles = 0
    
    # 初始次品率估计
    cipinlv1 = np.array([gujicipinlv(0.1, confidence_level) for _ in range(len(parts_df))])
    cipinlv2 = np.array([gujicipinlv(0.1, confidence_level) for _ in range(len(semi_df))])
    cipinlv3 = gujicipinlv(0.1, confidence_level)
    
    jiancelishi = {'parts': [], 'semi': [], 'final': []}
    juecelicheng = []
    
    while cycles < shoumaicishu and (cipinlv1.max() > threshold or cipinlv2.max() > threshold or cipinlv3 > threshold):
        cycles += 1
        # 记录次品率
        jiancelishi['parts'].append(cipinlv1.copy())
        jiancelishi['semi'].append(cipinlv2.copy())
        jiancelishi['final'].append(cipinlv3)
        
        # 计算零配件检测成本
        jiancechengben11 = 0
        jiancec11 = []
        for i, row in parts_df.iterrows():
            if row['检测成本'] < cipinlv1[i] * row['购买单价']:
                jiancechengben11 += row['检测成本']
                jiancec11.append(1)  # 进行检测
            else:
                jiancechengben11 += cipinlv1[i] * row['购买单价']
                jiancec11.append(0)  # 不检测
        bufenchengben11 += jiancechengben11
        
        # 计算半成品检测成本
        banchengpinchengben = 0
        benchengpinfenxi1 = []
        for i, row in semi_df.iterrows():
            if row['检测成本'] < cipinlv2[i] * (row['装配成本'] + row['拆解费用']):
                banchengpinchengben += row['检测成本']
                benchengpinfenxi1.append(1)
            else:
                banchengpinchengben += cipinlv2[i] * (row['装配成本'] + row['拆解费用'])
                benchengpinfenxi1.append(0)
        bufenchengben11 += banchengpinchengben
        
        # 计算成品检测成本
        zuizhonghuafei = 0
        if zuizhongjuedingshuchu.loc[0, '检测成本'] < cipinlv3 * (zuizhongjuedingshuchu.loc[0, '装配成本'] + zuizhongjuedingshuchu.loc[0, '市场售价'] + zuizhongjuedingshuchu.loc[0, '拆解费用']):
            zuizhonghuafei += zuizhongjuedingshuchu.loc[0, '检测成本']
            final_decision = 1
        else:
            zuizhonghuafei += cipinlv3 * (zuizhongjuedingshuchu.loc[0, '装配成本'] + zuizhongjuedingshuchu.loc[0, '市场售价'] + zuizhongjuedingshuchu.loc[0, '拆解费用'])
            final_decision = 0
        bufenchengben11 += zuizhonghuafei
        

        juecelicheng.append({
            'cycle': cycles,
            '零配件检测决策': jiancec11,
            '半成品检测决策': benchengpinfenxi1,
            '成品检测决策': final_decision
        })
        

        cipinlv1 *= (1 - dijianbili)
        cipinlv2 *= (1 - dijianbili)
        cipinlv3 *= (1 - dijianbili)
    
    return bufenchengben11, cycles, juecelicheng, jiancelishi


bufenchengben11_95, cycles_95, decisions_95, defects_95 = calculate_bufenchengben11(parts_df, semi_df, zuizhongjuedingshuchu, confidence_level=0.95)
bufenchengben11_90, cycles_90, decisions_90, defects_90 = calculate_bufenchengben11(parts_df, semi_df, zuizhongjuedingshuchu, confidence_level=0.90)

# 输出结果
def print_results(confidence, bufenchengben11, cycles, decisions, defects):
    print(f"\n{confidence}% 置信度下的决策分析结果：")
    print("=" * 40)
    print(f"单个产品总成本: {bufenchengben11:.2f}")
    print(f"总循环次数: {cycles}")
    print("=" * 40)

    print("决策详情：")
    print("-" * 40)

    for decision in decisions:
        print(f"第 {decision['cycle']} 次循环：")
        print(f"  零配件检测决策: {decision['零配件检测决策']}")
        print(f"  半成品检测决策: {decision['半成品检测决策']}")
        print(f"  成品检测决策: {decision['成品检测决策']}")
        print("-" * 40)

    print("=" * 40)


print_results(95, bufenchengben11_95, cycles_95, decisions_95, defects_95)
print_results(90, bufenchengben11_90, cycles_90, decisions_90, defects_90)


def plot_jiancelishi(defects, confidence):
    cycles = range(1, len(defects['parts']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, np.array(defects['parts'])[:, 0], label='零配件1次品率', color='orange', marker='o')
    plt.plot(cycles, np.array(defects['parts'])[:, 1], label='零配件2次品率', color='yellow', marker='o')
    plt.plot(cycles, np.array(defects['parts'])[:, 2], label='零配件3次品率', color='red', marker='o')
    plt.plot(cycles, np.array(defects['parts'])[:, 3], label='零配件4次品率', color='blue', marker='o')
    plt.plot(cycles, np.array(defects['parts'])[:, 4], label='零配件5次品率', color='black', marker='o')
    plt.plot(cycles, np.array(defects['parts'])[:, 5], label='零配件6次品率', color='green', marker='o')
    plt.plot(cycles, np.array(defects['semi'])[:, 0], label='半成品次品率', color='purple', marker='o')
    plt.plot(cycles, defects['final'], label='成品次品率', color='pink', marker='o')
    plt.xlabel('循环次数')
    plt.ylabel('次品率')
    plt.title(f'{confidence}% 置信度下的次品率历史')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_jiancelishi(defects_95, 95)
plot_jiancelishi(defects_90, 90)
