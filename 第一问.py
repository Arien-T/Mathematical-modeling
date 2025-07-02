import os
import sys
import subprocess
from scipy.stats import binom
import warnings


sys.stderr = open(os.devnull, 'w')
warnings.filterwarnings("ignore")


def install_packages(packages, mirror):
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-i', mirror])

install_packages(['seaborn', 'scipy'], 'https://pypi.tuna.tsinghua.edu.cn/simple')


zhixinshuiping = [0.95, 0.90]
jaincelv = 0.10

def jisuanlv(sample_size, zhixinshuiping, jaincelv):
    jieshoushuiping = binom.ppf(1-zhixinshuiping[1], sample_size, jaincelv)
    jujueshuiping = binom.ppf(zhixinshuiping[0], sample_size, jaincelv)
    return jieshoushuiping, jujueshuiping


yangbendaxiao1 = 100
yangbendaxiao2, jujuelv22 = jisuanlv(yangbendaxiao1, zhixinshuiping, jaincelv)

print("第一次检测：")
print(f"如果检测到次品数量大于{jujuelv22:.0f},则拒收")
print(f"如果检测到次品数量小于{yangbendaxiao2:.0f},则接收")


second_sample_size = 350
second_accept_threshold, second_reject_threshold = jisuanlv(second_sample_size, zhixinshuiping, jaincelv)

print("\n第二次检测：")
print(f"如果检测到次品数量大于{second_reject_threshold:.0f},则拒收")
print(f"如果检测到次品数量小于{second_accept_threshold:.0f},则接收")
