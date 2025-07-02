import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  


shoumaicishu = 10  
konu = 0.02   #次品率阈值
cipinlvjiangdi = 0.6

# 计算总成本函数
def jisuanzongchengben (df, defects, detection_col, loss_func):
    zonghuafei1 = 0
    juece = [] #决策方案
    
    for i in range(len(df)):
        if df.loc[i, '检测成本'] < loss_func(i, defects):
            zonghuafei1 += df.loc[i, '检测成本']
            juece.append(1)  # 1表示检测
        else:
            zonghuafei1 += loss_func(i, defects)
            juece.append(0)  # 0表示不检测
    return zonghuafei1, juece

# 计算零配件、半成品和成品的总成本
def jisuanzonghuafei11(bujia1, df_halfparts, df_final, shoumaicishu=shoumaicishu, cipinlvjiangdi=cipinlvjiangdi, konu=konu):
    zonghuafei1 = 0
    cycle = 0  # 售卖次数
    
    jaince11 = bujia1['次品率'].copy()
    halfjaince11 = df_halfparts['次品率'].copy()
    zongjaince11 = df_final['次品率'][0]
    
    juece = {'售卖次数': [], '零配件检测决策': [], '半成品检测决策': [], '成品检测决策': []}

    while cycle < shoumaicishu and (jaince11.max() > konu or halfjaince11.max() > konu or zongjaince11 > konu):
        cycle += 1
        
        # 计算零配件的检测成本
        lingpeijianjiance1dd = lambda i, defects: defects[i] * bujia1.loc[i, '购买单价']
        bufenhaufeia, bujia2 = jisuanzongchengben (bujia1, jaince11, '检测成本', lingpeijianjiance1dd)
        zonghuafei1 += bufenhaufeia
        
        # 计算半成品的检测成本
        banchengben55 = lambda i, defects: defects[i] * (df_halfparts.loc[i, '装配成本'] + df_halfparts.loc[i, '拆解费用'])
        banchengben66, halfpart_juece = jisuanzongchengben (df_halfparts, halfjaince11, '检测成本', banchengben55)
        zonghuafei1 += banchengben66
        
        # 计算成品的检测成本
        zongdajianchegben = zongjaince11 * (df_final.loc[0, '装配成本'] + df_final.loc[0, '市场售价'] + df_final.loc[0, '拆解费用'])
        if df_final.loc[0, '检测成本'] < zongdajianchegben:
            zonghuafei1 += df_final.loc[0, '检测成本']
            wupinjuece = 1
        else:
            zonghuafei1 += zongdajianchegben
            wupinjuece = 0
        

        juece['售卖次数'].append(cycle)
        juece['零配件检测决策'].append(bujia2)
        juece['半成品检测决策'].append(halfpart_juece)
        juece['成品检测决策'].append(wupinjuece)
        

        jaince11 *= (1 - cipinlvjiangdi)
        halfjaince11 *= (1 - cipinlvjiangdi)
        zongjaince11 *= (1 - cipinlvjiangdi)
    
    return zonghuafei1, cycle, juece


data1_0parts = {
    '零配件': ['零配件1', '零配件2', '零配件3', '零配件4', '零配件5', '零配件6', '零配件7', '零配件8'],
    '次品率': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    '购买单价': [2, 8, 12, 2, 8, 12, 8, 12],
    '检测成本': [1, 1, 2, 1, 1, 2, 1, 2]
}

data2_halfparts = {
    '半成品': ['半成品1', '半成品2', '半成品3'],
    '次品率': [0.10, 0.10, 0.10],
    '装配成本': [8, 8, 8],
    '检测成本': [4, 4, 4],
    '拆解费用': [6, 6, 6]
}

data3_finalproducts = {
    '成品': ['成品'],
    '次品率': [0.10],
    '装配成本': [8],
    '检测成本': [6],
    '拆解费用': [10],
    '市场售价': [200],
    '调换损失': [40]
}


shuchuges1 = pd.DataFrame(data1_0parts)
shuchuges2 = pd.DataFrame(data2_halfparts)
shuchuges3 = pd.DataFrame(data3_finalproducts)

# 计算总成本
zonghuafei1, cycle_count, decision_results = jisuanzonghuafei11(shuchuges1, shuchuges2, shuchuges3)

# 输出结果
print(f"单个成品总成本: {zonghuafei1:.2f}")
print(f"总售卖次数: {cycle_count}")
for cycle in range(cycle_count):
    print(f"第 {decision_results['售卖次数'][cycle]} 次循环：")
    print(f"  是否检测零配件: {decision_results['零配件检测决策'][cycle]}")
    print(f"  是否检测半成品: {decision_results['半成品检测决策'][cycle]}")
    print(f"  是否检测成品: {decision_results['成品检测决策'][cycle]}")


def ziongjueceshuchu11(bujia2, bujia1, cycles_labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用'coolwarm'色图，并设置合适的色标范围
    cmap = plt.get_cmap('OrRd')
    im = ax.imshow(bujia2.T, cmap=cmap, vmin=0, vmax=1)
    
    # 添加标题和标签
    ax.set_title("是否检测零配件", fontsize=16, fontweight='bold')
    ax.set_xticks(np.arange(len(cycles_labels)))
    ax.set_xticklabels(cycles_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.set_yticks(np.arange(len(bujia1)))
    ax.set_yticklabels(bujia1['零配件'], fontsize=12, fontweight='bold')
    ax.set_ylabel('(0不检测，1检测)', fontsize=14, fontweight='bold')
    

    for i in range(len(bujia1)):
        for j in range(len(bujia2)):
            value = bujia2[j, i]
            text = ax.text(j, i, f'{value:.2f}',
                           ha="center", va="center", color="white" if value > 0.5 else "black",
                           fontsize=10, fontweight='bold')
    

    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('检测概率', fontsize=14, fontweight='bold')
    

    ax.grid(which='both', color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np

def ziongjueceshuchu22(banpart_juece_juzhen, final_product_juece, cycle_labels):
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(cycle_labels))
    width = 0.35


    rects1 = ax.bar(x - width/2, np.sum(banpart_juece_juzhen, axis=1), width, 
                    color=plt.cm.viridis(np.linspace(0.2, 0.8, len(cycle_labels))),
                    edgecolor='black', alpha=0.8, label='检测半成品')
    

    ax.plot(x, final_product_juece, color='blue', marker='o', markersize=10, 
            markerfacecolor='cyan', markeredgewidth=2, markeredgecolor='blue', 
            linestyle='-', linewidth=2, alpha=0.7, label='检测成品')


    ax.set_ylabel('(0不检测, 1检测)', fontsize=14, fontweight='bold')
    ax.set_title('是否进行半成品和成品检测', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cycle_labels, rotation=45, fontsize=12)
    

    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_facecolor('#f7f7f7')
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, fancybox=True)


    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

def ziongjueceshuchu33(bujia1, df_halfparts, df_final, cycle_count):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算每个周期的总成本
    cost_per_cycle = [jisuanzonghuafei11(bujia1, df_halfparts, df_final, shoumaicishu=i+1)[0] for i in range(cycle_count)]
    

    ax.plot([f"售卖次数 {i+1}" for i in range(cycle_count)], cost_per_cycle, marker='o', color='darkblue', linestyle='-', linewidth=2, markersize=8, label='单个产品总成本')


    ax.set_title('单个产品总成本变化', fontsize=16, fontweight='bold')
    ax.set_xlabel('售卖次数', fontsize=14, fontweight='bold')
    ax.set_ylabel('单个成品总成本', fontsize=14, fontweight='bold')
    

    ax.grid(True, linestyle='--', color='gray', alpha=0.6)
    

    ax.legend(fontsize=12)


    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    plt.show()

part_juece_juzhen = np.array(decision_results['零配件检测决策'])
banpart_juece_juzhen = np.array(decision_results['半成品检测决策'])
final_product_juece = decision_results['成品检测决策']

cycle_labels = [f"售卖次数 {i+1}" for i in range(cycle_count)]

ziongjueceshuchu11(part_juece_juzhen, shuchuges1, cycle_labels)
ziongjueceshuchu22(banpart_juece_juzhen, final_product_juece, cycle_labels)
ziongjueceshuchu33(shuchuges1, shuchuges2, shuchuges3, cycle_count)
