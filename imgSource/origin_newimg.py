import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, cm
import json

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline, make_interp_spline
import matplotlib.colors as mcolors


def Fig13():
    fontsize = 50
    y0_values = [0.99206, 0.99404, 1.00000, 0.99504, 0.99899, 1.00000, 0.99899]
    y1_values = [0.96790, 0.98132, 0.99899, 0.99498, 0.96899, 1.00000, 0.98776]
    y2_values = [0.98322, 0.98615, 1.00000, 0.99501, 0.98030, 1.00000, 0.99788]

    # patterns = ('-', ' ', '\\', '\\', '*', 'o', 'O', '.')  # 设置不同柱状图的填充样式
    categories = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI', 'C&C']
    values = np.array([y0_values, y1_values, y2_values])
    markers = ['v', 'o', '^', 's', 'd', '*']
    # 颜色设置

    lables = ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE']
    # 计算柱子宽度
    bar_width = 0.35
    indices = np.arange(len(categories))
    plt.rc('font', family='Times New Roman')
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.grid(ls='--', alpha=0.5)
    # 绘制每个类别的四个柱子和误差条
    for i, category in enumerate(lables):
        ax.bar(indices + (i - 1) * (bar_width / 1.5), values[i], bar_width, color=colors[i], label=category, alpha=0.3)

    # 添加曲线
    for i, category in enumerate(lables):
        # 生成平滑曲线的数据点
        x_smooth = np.linspace(indices.min(), indices.max() + bar_width * (len(lables) - 1), 50)
        spline = make_interp_spline(indices + i * bar_width, values[i], k=2)  # k=3 表示三次样条插值
        y_smooth = spline(x_smooth)

        ax.plot(x_smooth, y_smooth, linestyle='-', color=colors[i], linewidth=4, label=category)

    # 图例设置
    ax.legend(lables, loc='upper left', bbox_to_anchor=(0, 1.22), ncol=2, fontsize=fontsize - 15, frameon=False,
              shadow=False, fancybox=True, edgecolor='black', facecolor='white')

    # 添加标签和标题
    ax.set_xlabel('Flow Type', fontsize=fontsize, family='Times New Roman')
    ax.set_ylabel('Accuracy', fontsize=fontsize, family='Times New Roman')
    # ax.set_title('Grouped Bar Chart with Asymmetric Error Bars')

    # 调整X轴刻度标签
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, fontsize=fontsize - 15)
    ax.set_xlim(-0.5, 6.5)
    # 调整Y轴刻度标签
    ax.set_ylim(0.93, 1.01)
    ax.tick_params(axis='y', labelsize=fontsize - 10)
    # 显示图表
    plt.tight_layout()
    plt.savefig(f"../img/fig13.png", dpi=300)
    plt.savefig(f"../img/fig13.pdf", dpi=600)
    # 显示图表
    plt.show()

def Fig37():
    fontsize = 50
    y0_values = [85.13, 340.50, 43.28, 1.1303]
    y1_values = [7.13, 28.53, 114.98, 0.7146]
    y2_values = [7.13, 28.53, 114.98, 0.7146]

    patterns = ('-', ' ', '\\', '\\', '*', 'o', 'O', '.')  # 设置不同柱状图的填充样式
    categories = ['Parameters\n($M$)', 'Size\n($MB$)', 'Speed\n(samples/$s$)', 'Detection Cost\n($\%$)']
    values = np.array([y0_values, y1_values, y2_values])
    markers = ['v', 'o', '^', 's', 'd', '*']
    # 颜色设置
    colors = ['#5760A8', '#F6B816', '#DE4849']
    lables = ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE']
    # 计算柱子宽度
    bar_width = 0.2
    indices = np.arange(len(categories))
    plt.rc('font', family='Times New Roman')
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(ls='--', alpha=0.5)
    plt.rc('font', family='Times New Roman')
    # 绘制每个类别的四个柱子和误差条
    for i, category in enumerate(lables):
        bars = ax.bar(indices + i * bar_width, values[i], bar_width, color=colors[i], label=category)
    # 柱子上添加数值
    for i, category in enumerate(lables):
        if i != 1:
            for j, value in enumerate(values[i]):
                ax.text(j + i * bar_width, value + 0.1, f"{value:.2f}", ha='center', va='bottom',
                        fontsize=fontsize - 15)
    # 添加曲线
    # for i, category in enumerate(lables):
    #     ax.plot(indices + i * bar_width, values[i],  linestyle='-', color=colors[i], linewidth=3, marker=markers[i], markeredgewidth=3, markeredgecolor='black',
    #             markerfacecolor='white', markersize=15, label=category)


    # 添加标签和标题
    ax.set_xlabel('Metrics', fontsize=fontsize, family='Times New Roman')
    ax.set_ylabel('Value', fontsize=fontsize, family='Times New Roman')
    # ax.set_title('Grouped Bar Chart with Asymmetric Error Bars')

    # 调整X轴刻度标签
    ax.set_xticks(indices + bar_width * 1.5)
    ax.set_xticklabels(categories, fontsize=fontsize - 15)
    # Y轴调整为对数坐标
    ax.set_yscale('log')
    # Y轴范围0-400
    ax.set_ylim(0, 600)
    ax.tick_params(axis='y', labelsize=fontsize - 10)
    # # 添加统一的图例
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
    # fig.legend(handles, lables, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=fontsize - 15,
    #            handlelength=0.5, handleheight=0.4)  # 调整图例中颜色块的大小
    # # 调整图表布局,把图例放在子图之下
    plt.tight_layout()
    plt.savefig(f"../img/fig37.png", dpi=300)
    plt.savefig(f"../img/fig37.pdf", dpi=600)
    # 显示图表
    plt.show()
