import numpy as np
import matplotlib.pyplot as plt
from itertools import product


from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


cases = [
    {'零配件1次品率': 0.1, '零配件1购买单价': 4, '零配件1检测成本': 2, '零配件2次品率': 0.1, '零配件2购买单价': 18,
     '零配件2检测成本': 3, '成品次品率': 0.1, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 6, '拆解费用': 5},
    {'零配件1次品率': 0.2, '零配件1购买单价': 4, '零配件1检测成本': 2, '零配件2次品率': 0.2, '零配件2购买单价': 18,
     '零配件2检测成本': 3, '成品次品率': 0.2, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 6, '拆解费用': 5},
    {'零配件1次品率': 0.1, '零配件1购买单价': 4, '零配件1检测成本': 2, '零配件2次品率': 0.1, '零配件2购买单价': 18,
     '零配件2检测成本': 3, '成品次品率': 0.1, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 30, '拆解费用': 5},
    {'零配件1次品率': 0.2, '零配件1购买单价': 4, '零配件1检测成本': 1, '零配件2次品率': 0.2, '零配件2购买单价': 18,
     '零配件2检测成本': 1, '成品次品率': 0.2, '成品检测成本': 2, '装配成本': 6, '市场售价': 56, '调换损失': 30, '拆解费用': 5},
    {'零配件1次品率': 0.1, '零配件1购买单价': 4, '零配件1检测成本': 8, '零配件2次品率': 0.2, '零配件2购买单价': 18,
     '零配件2检测成本': 1, '成品次品率': 0.1, '成品检测成本': 2, '装配成本': 6, '市场售价': 56, '调换损失': 10, '拆解费用': 5},
    {'零配件1次品率': 0.05, '零配件1购买单价': 4, '零配件1检测成本': 2, '零配件2次品率': 0.05, '零配件2购买单价': 18,
     '零配件2检测成本': 3, '成品次品率': 0.05, '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 10, '拆解费用': 40}
]

# 计算总成本、合格率和调换成本的函数
def zongjisuan(case, detect_parts1, detect_parts2, detect_final, dismantle):
    N = 100  # 固定生产的成品数
    
    # 购买零配件数量
    peijian1 = N / (1 - case['零配件1次品率'] * (1 - detect_parts1))
    peijian2 = N / (1 - case['零配件2次品率'] * (1 - detect_parts2))
    
    # 成本1：购买成本
    cost_purchase = peijian1 * case['零配件1购买单价'] + peijian2 * case['零配件2购买单价']
    
    # 成本2：零配件1和零配件2检测成本
    jiancechengben1 = peijian1 * case['零配件1检测成本'] * detect_parts1
    jiancechengben2 = peijian2 * case['零配件2检测成本'] * detect_parts2
    
    # 成本3: 装配成本
    zhuangpeichengben = N * case['装配成本']
    
    # 成本4: 成品检测成本
    jiancechengben3 = N * case['成品检测成本'] * detect_final
    
    # 成本5: 调换成本
    diaohuanchengben = (1 - detect_final) * N * case['成品次品率'] * case['调换损失']
    
    # 成本6: 拆解成本
    chaijiechengben = N * case['成品次品率'] * dismantle * (case['拆解费用'] - 0.6 * case['市场售价'])
    
    # 总成本
    zongchengben = cost_purchase + jiancechengben1 + jiancechengben2 + zhuangpeichengben + jiancechengben3 + diaohuanchengben + chaijiechengben
    
    # 合格率
    cipinlv1 = 1 - case['零配件1次品率'] * (1 - detect_parts1)
    cipinlv2 = 1 - case['零配件2次品率'] * (1 - detect_parts2)
    cipinlv3 = cipinlv1 * cipinlv2 * (1 - case['成品次品率'] * (1 - detect_final))
    
    return zongchengben, cipinlv3, diaohuanchengben

# 生成策略
strategies = list(product([1, 0], repeat=4))  # (detect_parts1, detect_parts2, detect_final, dismantle)


def celuejieguoshuchu(celue1):
    detect_parts1, detect_parts2, detect_final, dismantle = strategies[celue1]
    explanation = f"策略 {celue1 + 1}: "
    explanation += "检测零配件 1" if detect_parts1 else "不检测零配件 1"
    explanation += "，检测零配件 2" if detect_parts2 else "，不检测零配件 2"
    explanation += "，检测成品" if detect_final else "，不检测成品"
    explanation += "，拆解不合格成品" if dismantle else "，不拆解不合格成品"
    return explanation


costs = []
defective_rates = []
exchanges = []

for case_idx, case in enumerate(cases):
    print(f"\n情况 {case_idx + 1}:")
    
    qingkuanghuafei = []
    qingkuangcipinlv = []
    jiaohuanqingkuang = []
    
    for celue1, strategy in enumerate(strategies):
        zongchengben, cipinlv3, exchange = zongjisuan(case, *strategy)
        
        qingkuanghuafei.append(zongchengben)
        qingkuangcipinlv.append(cipinlv3)
        jiaohuanqingkuang.append(exchange)
        
        print(f"{celuejieguoshuchu(celue1)}: 总成本 = {zongchengben:.2f}, 调换成本 = {exchange:.2f}, 成品合格率 = {cipinlv3:.2f}")
    
    costs.append(qingkuanghuafei)
    defective_rates.append(qingkuangcipinlv)
    exchanges.append(jiaohuanqingkuang)


costs = np.array(costs)
defective_rates = np.array(defective_rates)
exchanges = np.array(exchanges)


plt.figure(figsize=(15, 8))
colors = plt.cm.viridis(np.linspace(0, 1, costs.shape[1]))  # 使用渐变色
for i in range(costs.shape[1]):
    plt.plot(range(1, costs.shape[0] + 1), costs[:, i], marker='o', linestyle='-.', linewidth='2',
             color=colors[i], label=f"策略 {i + 1} - 总成本", alpha=0.8)

plt.xlabel("情况", fontsize=14, fontweight='bold')
plt.ylabel("总成本", fontsize=14, fontweight='bold')
plt.title("不同情况和策略下的成本比较", fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()
