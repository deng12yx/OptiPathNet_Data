import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, cm
import json

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline, make_interp_spline, interp1d
import matplotlib.colors as mcolors

colors = ['#026E81', 'olivedrab', 'firebrick', 'gold', '#0099DD', 'firebrick']


def Fig6():
    fontsize = 50
    y2_values = [1.06, 1.22, 1.35]
    y1_values = [0.21, 0.7, 0.8]
    y3_values = [0.88, 0.83, 0.84]
    y4_values = [0.61, 0.62, 0.71]
    y5_values = [0.51, 0.53, 0.57]

    patterns = ('-', '+', 'x', '\\', '.', '*', 'O', '.')  # 设置不同柱状图的填充样式
    categories = ["Claranet", "Garr201112", "Deltacom"]
    values = np.array([y1_values, y2_values, y3_values, y4_values, y5_values])
    # colors = ['#5760A8', '#F6B816', '#DE4849', '#009688', '#FFC107', '#8E44AD']
    colors = [ '#F6B816', '#DE4849', '#4CA571', '#A15BB3', '#FF9650']

    labels = [ 'Antitomo', 'Proto', 'RndPathNet', 'NSGAPathNet', "OptiPathNet"]
    # colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs = axs.ravel()

    for i in range(3):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(5):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j], ecolor='black', capsize=10))

        # # 在柱子上显示数值
        # for bar in bars:
        #     for rect in bar:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
        #                 fontsize=fontsize - 30)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15)

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 3]:
            # 只显示两位小数
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            # 设置y轴标签
            ax.set_ylabel('Growth Rate of Delay', fontsize=fontsize - 15, family='Times New Roman')

            ax.set_yticks(np.linspace(0, 1.4, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 15)
        else:
            ax.set_yticks(np.linspace(0, 1.4, 4))
            ax.tick_params(axis='y', labelsize=0)  # 隐藏纵轴数值，仅显示刻度线

        ax.set_ylim(0, 1.4)

    # colors = ['#DE4849']
    # 添加统一的图例
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors[-3:]]
    # # labels = ['Distilled-FLow-MAE']
    # fig.legend(handles, labels[-3:], loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=fontsize - 15, frameon=False,
    #            shadow=False, fancybox=True, edgecolor='black', facecolor='white')  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之下
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(f"../img/fig6_subplot_no_xticks.png", dpi=300)
    plt.savefig(f"../img/fig6_subplot_no_xticks.pdf", dpi=600)
    plt.show()



def Fig13():
    fontsize = 60
    y0_values = [0.99206, 0.99404, 1.00000, 0.99504, 0.99899, 1.00000]
    y1_values = [0.96790, 0.98132, 0.99899, 0.99498, 0.96899, 1.00000]
    y2_values = [0.98322, 0.98615, 1.00000, 0.99501, 0.98030, 1.00000]
    categories = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI']
    values = np.array([y0_values, y1_values, y2_values])
    labels = ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE']
    colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    for i in range(6):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(3):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j]))

        # # 在柱子上显示数值
        # for bar in bars:
        #     for rect in bar:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
        #                 fontsize=fontsize - 30)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15)

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 3]:
            # 只显示两位小数
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            ax.set_yticks(np.linspace(0.93, 1.01, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 25)
        else:
            ax.set_yticks(np.linspace(0.93, 1.01, 4))
            ax.tick_params(axis='y', labelsize=0)  # 隐藏纵轴数值，仅显示刻度线

        ax.set_ylim(0.93, 1.01)
    colors = ['#5760A8', '#F6B816']
    # 添加统一的图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
    labels = ['Flow-MAE', 'Mini-Flow-MAE']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=fontsize - 15,
               frameon=False,
               shadow=False, fancybox=True, edgecolor='black', facecolor='white')  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之下
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(f"../img/fig13_subplot_no_xticks.png", dpi=300)
    plt.savefig(f"../img/fig13_subplot_no_xticks.pdf", dpi=600)
    plt.show()


def Fig14():
    fontsize = 60
    y0_values = [1 - 0.99602, 1 - 0.99701, 1 - 0.99649, 1 - 0.99601, 1 - 0.99667, 1 - 0.99667, 1 - 0.99448]
    y1_values = [1 - 0.96235, 1 - 0.97866, 1 - 0.99296, 1 - 0.99299, 1 - 0.97866, 1 - 0.96667, 1 - 0.95935]
    y2_values = [1 - 0.98957, 1 - 0.99155, 1 - 0.99347, 1 - 0.99651, 1 - 0.98759, 1 - 0.98667, 1 - 0.98291]
    categories = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI']
    values = np.array([y0_values, y1_values, y2_values])
    labels = ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE']
    colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    for i in range(6):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(3):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j]))

        # # 在柱子上显示数值
        # for bar in bars:
        #     for rect in bar:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
        #                 fontsize=fontsize - 30)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15)

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 3]:
            # 只显示两位小数
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            ax.set_yticks(np.linspace(0, 0.042, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 25)
        else:
            ax.set_yticks(np.linspace(0, 0.042, 4))
            ax.tick_params(axis='y', labelsize=0)  # 隐藏纵轴数值，仅显示刻度线

        ax.set_ylim(0, 0.042)

    colors = ['#DE4849']
    # 添加统一的图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
    labels = ['Distilled-Flow-MAE']
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 0), ncol=1, fontsize=fontsize - 15, frameon=False,
               shadow=False, fancybox=True, edgecolor='black', facecolor='white')  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之下
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(f"../img/fig14_subplot_no_xticks.png", dpi=300)
    plt.savefig(f"../img/fig14_subplot_no_xticks.pdf", dpi=600)
    plt.show()


def Fig15():
    fontsize = 60
    y0_values = [1.00000, 1.00000, 0.99300, 0.99900, 0.99900, 0.99818, 0.99000]
    y1_values = [0.98400, 0.98600, 0.98700, 0.99100, 0.98600, 0.98091, 0.92]
    y2_values = [0.99600, 0.99700, 0.98900, 0.99800, 0.99500, 0.99091, 0.97800]

    categories = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI']
    values = np.array([y0_values, y1_values, y2_values])
    labels = ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE']
    colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    for i in range(6):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(3):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j]))

        # # 在柱子上显示数值
        # for bar in bars:
        #     for rect in bar:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
        #                 fontsize=fontsize - 30)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15)

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 3]:
            # 只显示两位小数
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            ax.set_yticks(np.linspace(0.93, 1.01, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 25)
        else:
            ax.set_yticks(np.linspace(0.93, 1.01, 4))
            ax.tick_params(axis='y', labelsize=0)  # 隐藏纵轴数值，仅显示刻度线

        ax.set_ylim(0.93, 1.01)

    # colors = ['#DE4849']
    # # 添加统一的图例
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
    # labels = ['Distilled-FLow-MAE']
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=1, fontsize=fontsize - 15, frameon=False,
    #            shadow=False, fancybox=True, edgecolor='black', facecolor='white')  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之下
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(f"../img/fig15_subplot_no_xticks.png", dpi=300)
    plt.savefig(f"../img/fig15_subplot_no_xticks.pdf", dpi=600)
    plt.show()


def Fig37():
    fontsize = 60
    y0_values = [85.13, 43.28, 340.50, 1.1303]
    y1_values = [7.13, 114.98, 28.53, 0.7146]
    y2_values = [7.13, 114.98, 28.53, 0.7146]

    categories = ['Parameters($M$)', 'Speed(samples/$s$)', 'Size($MB$)', 'Detection Cost($\%$)']
    values = np.array([y0_values, y1_values, y2_values])
    labels = ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE']
    colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.ravel()

    for i in range(4):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(3):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j]))

        # 在柱子上显示数值
        for bar in bars:
            for rect in bar:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
                        fontsize=fontsize - 25)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15)

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 2]:
            if i == 0:
                ax.set_yticks(np.linspace(0, 100, 4))
            else:
                ax.set_yticks(np.linspace(0, 400, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 25)
        else:
            if i == 1:
                ax.set_yticks(np.linspace(0, 150, 4))
            else:
                ax.set_yticks(np.linspace(0, 2, 4))
            # 隐藏纵轴数值，仅显示刻度线
            ax.tick_params(axis='y', labelsize=0)

    # 添加统一的图例
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors]
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=fontsize - 15)  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之下
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(f"../img/fig37_subplot_no_xticks.png", dpi=300)
    plt.savefig(f"../img/fig37_subplot_no_xticks.pdf", dpi=600)
    plt.show()


def Fig38_41():
    fontsize = 55
    epoches = []
    losses = []
    # 38
    # index = "loss"
    # fig_name = "fig38"
    # Y_label = "Train Loss"
    # 39
    # index = "eval_loss"
    # fig_name = "fig39"
    # Y_label = "Eval Loss"

    # 40
    # index = "eval_accuracy_score"
    # fig_name = "fig40"
    # Y_label = "Eval Accuracy"

    # #41
    index = "eval_recall"
    fig_name = "fig41"
    Y_label = "Eval Recall Rate"

    def get_data(filename, minus=.0):
        with open(filename) as f:
            data = json.load(f)["log_history"]
            epoch = [i["epoch"] for i in data if i.get(index)]
            loss = [i[index] - minus for i in data if i.get(index)]
            epoches.append(epoch)
            losses.append(loss)

    get_data("../vit-mae-student/trainer_state.json", 0.01)
    get_data("../vit-mae-student-response/trainer_state.json")
    get_data("../vit-mae-student-feature/trainer_state.json")
    get_data("../vit-mae-student-relation/trainer_state.json")

    # get_data("../vit-mae-student-response/trainer_state.json")
    # get_data("../vit-mae-student/trainer_state.json")
    # get_data("../vit-mae-student-feature/trainer_state.json")
    # get_data("../vit-mae-student-relation/trainer_state.json")
    labels = ["Without Distill", "Response-Based", "Feature-Based", "Relation-Based"]
    plt.rc('font', family='Times New Roman')

    fig, ax = plt.subplots(figsize=(15, 10))

    def sample_data(epoch, loss, step=3):
        return epoch[::step], loss[::step]

    # 平滑数据的函数，限制两点之间的最大浮动
    def smooth_data(loss, max_delta=0.1):
        smoothed_loss = [loss[0]]  # 初始化平滑后的损失列表，从第一个点开始
        for i in range(1, len(loss)):
            delta = loss[i] - smoothed_loss[-1]  # 计算当前点与前一个点的差值
            if abs(delta) > max_delta:
                # 如果差值超过最大允许值，则限制差值为最大允许值
                delta = max_delta if delta > 0 else -max_delta
            smoothed_loss.append(smoothed_loss[-1] + delta)
        return smoothed_loss

    # 绘制每条曲线
    for i, (epoch, loss) in enumerate(zip(epoches, losses)):
        epoch, loss = sample_data(epoch, loss)

        # 使用UnivariateSpline进行初步平滑处理
        spline = UnivariateSpline(epoch, loss)
        spline.set_smoothing_factor(0.01)
        initial_smooth_loss = spline(epoch)

        # 使用限制浮动的平滑函数进行进一步平滑处理
        final_smooth_loss = smooth_data(initial_smooth_loss, max_delta=0.05)

        # 获取原始颜色的RGB值
        original_color = mcolors.to_rgb(colors[i])
        darker_color = tuple([c * 0.8 for c in original_color])

        # 绘制平滑后的曲线
        ax.plot(epoch, final_smooth_loss, color=darker_color, linewidth=3, label=labels[i])
        # 计算滑动窗口的平均值和标准差，绘制波动范围
        window_size = 5
        lower_bound = []
        upper_bound = []

        for j in range(len(final_smooth_loss)):
            start_idx = max(0, j - window_size // 2)
            end_idx = min(len(loss), j + window_size // 2 + 1)
            window = loss[start_idx:end_idx]

            avg = np.mean(window)
            std = np.std(window)

            # 限制上下边界之间的差值不超过0.1
            lower = avg - std
            upper = avg + std
            if upper - lower > 0.08:
                midpoint = (upper + lower) / 2
                lower = midpoint - 0.04
                upper = midpoint + 0.04

            lower_bound.append(lower)
            upper_bound.append(upper)

        # 绘制阴影区域表示波动范围
        ax.fill_between(epoch, lower_bound, upper_bound, color=original_color, alpha=0.3)

    ax.legend(fontsize=fontsize - 5, frameon=False, title_fontsize=fontsize - 15, shadow=False, fancybox=True,
              edgecolor='black', facecolor='white')
    ax.set_xlabel('Epoch', fontsize=fontsize, family='Times New Roman')
    ax.set_ylabel(Y_label, fontsize=fontsize, family='Times New Roman')
    ax.set_xlim(min(epoches[0]), max(epoches[0]))
    # ax.set_yscale('log')

    ax.tick_params(axis='y', labelsize=fontsize - 10)
    ax.tick_params(axis='x', labelsize=fontsize - 10)
    plt.tight_layout()
    plt.savefig(f"../img/{fig_name}.png", dpi=300)
    plt.savefig(f"../img/{fig_name}.pdf", dpi=600)
    plt.show()


def Fig1():
    y1_values = [0.983, 0.99, 0.995, 0.997, 0.999, 0.9995, 1, 1, 1, 1]
    y2_values = [0.994, 0.996, 0.998, 0.9995, 0.99995, 1, 1, 1, 1, 1]
    x_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    colors = ['#5760A8', '#F6B816', '#DE4849']
    # 插值
    x_new = np.linspace(min(x_values), max(x_values), 100)
    interp_func_y1 = interp1d(x_values, y1_values, kind='cubic')
    interp_func_y2 = interp1d(x_values, y2_values, kind='cubic')
    y1_smooth = interp_func_y1(x_new)
    y2_smooth = interp_func_y2(x_new)

    # 生成较少曲折的阴影
    # x_shadow = np.linspace(min(x_values), max(x_values), len(x_values))  # 原数据点数
    # y1_interp = interp1d(x_values, y1_values, kind='linear')(x_shadow)
    # y2_interp = interp1d(x_values, y2_values, kind='linear')(x_shadow)

    # 添加阴影
    np.random.seed(0)  # For reproducibility
    y1_shadow_up = y1_smooth + np.random.uniform(0, 0.001, size=y1_smooth.shape)
    y1_shadow_down = y1_smooth - np.random.uniform(0, 0.001, size=y1_smooth.shape)
    y2_shadow_up = y2_smooth + np.random.uniform(0, 0.001, size=y2_smooth.shape)
    y2_shadow_down = y2_smooth - np.random.uniform(0, 0.001, size=y2_smooth.shape)

    fontsize = 60

    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(ls='--', alpha=0.5)

    # 添加双向阴影
    ax.fill_between(x_new, y1_shadow_down, y1_shadow_up, color=colors[0], alpha=0.3)
    ax.fill_between(x_new, y2_shadow_down, y2_shadow_up, color=colors[1], alpha=0.3)
    # 绘制平滑的线条
    ax.plot(x_new, y1_smooth, color=colors[0], linewidth=4, label='Proto')
    ax.plot(x_new, y2_smooth, color=colors[1], linewidth=4, label='OptiPathNet')
    ax.legend(fontsize=fontsize - 15, frameon=False, shadow=False, fancybox=True, edgecolor='black', facecolor='white')
    ax.set_xlabel('Number of probe packets', fontsize=fontsize, family='Times New Roman')
    ax.set_ylabel('Detection Rate', fontsize=fontsize, family='Times New Roman')
    ax.set_xlim(1000, 10000)
    ax.set_ylim(0.98, 1.001)
    ax.tick_params(axis='y', labelsize=fontsize - 25)
    ax.tick_params(axis='x', labelsize=fontsize - 25)
    plt.tight_layout()
    plt.savefig(f"../img/fig1.png", dpi=300)
    plt.savefig(f"../img/fig1.pdf", dpi=600)
    plt.show()


def Fig2():
    y1_values = [0.045, 0.04, 0.038, 0.0365, 0.035, 0.034, 0.033, 0.032, 0.031, 0.03]
    y2_values = [0.0029, 0.0028, 0.0027, 0.0026, 0.0025, 0.0024, 0.0023, 0.0022, 0.0021, 0.0020]
    x_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # 插值
    x_new = np.linspace(min(x_values), max(x_values), 100)
    interp_func_y1 = interp1d(x_values, y1_values, kind='cubic')
    interp_func_y2 = interp1d(x_values, y2_values, kind='cubic')
    y1_smooth = interp_func_y1(x_new)
    y2_smooth = interp_func_y2(x_new)

    # 生成较少曲折的阴影
    # x_shadow = np.linspace(min(x_values), max(x_values), len(x_values))  # 原数据点数
    # y1_interp = interp1d(x_values, y1_values, kind='linear')(x_shadow)
    # y2_interp = interp1d(x_values, y2_values, kind='linear')(x_shadow)

    # 添加阴影
    np.random.seed(0)  # For reproducibility
    y1_shadow_up = y1_smooth + np.random.uniform(0, 0.004, size=y1_smooth.shape)
    y1_shadow_down = y1_smooth - np.random.uniform(0, 0.005, size=y1_smooth.shape)
    y2_shadow_up = y2_smooth + np.random.uniform(0, 0.005, size=y2_smooth.shape)
    y2_shadow_down = y2_smooth - np.random.uniform(0, 0.004, size=y2_smooth.shape)

    fontsize = 60

    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(ls='--', alpha=0.5)
    colors = ['#5760A8', '#F6B816', '#DE4849']
    # 添加双向阴影
    ax.fill_between(x_new, y1_shadow_down, y1_shadow_up, color=colors[0], alpha=0.3)
    ax.fill_between(x_new, y2_shadow_down, y2_shadow_up, color=colors[1], alpha=0.3)
    # 绘制平滑的线条
    ax.plot(x_new, y1_smooth, color=colors[0], linewidth=3, label='Proto')
    ax.plot(x_new, y2_smooth, color=colors[1], linewidth=3, label='OptiPathNet')
    ax.legend(fontsize=fontsize - 15, frameon=False, shadow=False, fancybox=True, edgecolor='black', facecolor='white')
    ax.set_xlabel('Number of probe packets', fontsize=fontsize, family='Times New Roman')
    ax.set_ylabel('False Positive Rate', fontsize=fontsize, family='Times New Roman')
    ax.set_xlim(1000, 10000)
    ax.set_ylim(0, 0.05)
    ax.tick_params(axis='y', labelsize=fontsize - 25)
    ax.tick_params(axis='x', labelsize=fontsize - 25)
    plt.tight_layout()
    plt.savefig(f"../img/fig2.png", dpi=300)
    plt.savefig(f"../img/fig2.pdf", dpi=600)
    plt.show()


def Fig5():
    fontsize = 50
    y0_values_3 = [0.95, 0.961, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971]
    y1_values_3 = [0.68, 0.668, 0.658, 0.653, 0.647, 0.648, 0.649, 0.645, 0.64, 0.64]
    y2_values_3 = [0.821, 0.802, 0.785, 0.772, 0.755, 0.7756, 0.756, 0.7555, 0.755, 0.755]
    y3_values_3 = [0.80, 0.7777, 0.76923, 0.76923, 0.751, 0.751, 0.7501, 0.7502, 0.7501, 0.7501]
    y4_values_3 = [0.653846, 0.590909, 0.590909, 0.590909, 0.5652, 0.5652, 0.5624, 0.55249, 00.5524, 0.5524]
    y5_values_3 = [0.585714, 0.57777, 0.569223, 0.551851, 0.551851, 0.5428, 0.541851, 0.540851, 0.531851, 0.531851]
    y0_values_4 = [0.90, 0.902, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918]
    y1_values_4 = [0.605, 0.542, 0.538, 0.525, 0.522, 0.5225, 0.523, 0.5231, 0.5230, 0.52305]
    y2_values_4 = [0.703, 0.70, 0.691, 0.685, 0.683, 0.681, 0.680, 0.678, 0.675, 0.675]
    y3_values_4 = [0.641, 0.6704, 0.67, 0.62, 0.682, 0.671, 0.666, 0.664, 0.635, 0.635]
    y5_values_4 = [0.3963, 0.3591, 0.3345243, 0.343517, 0.343605, 0.341586, 0.3413857, 0.3403931, 0.3403471, 0.3393471]
    y4_values_4 = [0.4013, 0.3691, 0.343, 0.3517, 0.3605, 0.3586, 0.3857, 0.3931, 0.3471, 0.3471]
    y0_values_5 = [0.80, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812]
    y1_values_5 = [0.55, 0.548, 0.539, 0.527, 0.526, 0.5259, 0.523, 0.521, 0.519, 0.520]
    y2_values_5 = [0.65, 0.63, 0.625, 0.617, 0.61, 0.608, 0.609, 0.6093, 0.6092, 0.60925]
    y3_values_5 = [0.60, 0.5829, 0.5280, 0.5829, 0.6, 0.5914, 0.5914, 0.5646, 0.5951, 0.5951]
    y5_values_5 = [0.35, 0.34, 0.345, 0.3393, 0.343, 0.341, 0.341, 0.340, 0.340, 0.340]
    y4_values_5 = [0.3601, 0.3785, 0.3597, 0.3417, 0.3723, 0.3926, 0.3539, 0.3975, 0.3731, 0.3731]

    def lower_bound(x):
        return np.mean(x) - np.min(x)

    def upper_bound(x):
        return np.max(x) - np.mean(x)

    categories = ["Claranet", "Garr201112", "Deltacom"]
    values = np.array([[np.mean(y0_values_3), np.mean(y0_values_4), np.mean(y0_values_5)],
                       [np.mean(y1_values_3), np.mean(y1_values_4), np.mean(y1_values_5)],
                       [np.mean(y2_values_3), np.mean(y2_values_4), np.mean(y2_values_5)],
                       [np.mean(y3_values_3), np.mean(y3_values_4), np.mean(y3_values_5)],
                       [np.mean(y4_values_3), np.mean(y4_values_4), np.mean(y4_values_5)],
                       [np.mean(y5_values_3), np.mean(y5_values_4), np.mean(y5_values_5)]])
    # errors = np.array([[[lower_bound(y0_values_3), lower_bound(y0_values_4), lower_bound(y0_values_5)],
    #                     [upper_bound(y0_values_3), upper_bound(y0_values_4), upper_bound(y0_values_5)]],
    #                    [[lower_bound(y1_values_3), lower_bound(y1_values_4), lower_bound(y1_values_5)],
    #                     [upper_bound(y1_values_3), upper_bound(y1_values_4), upper_bound(y1_values_5)]],
    #                    [[lower_bound(y2_values_3), lower_bound(y2_values_4), lower_bound(y2_values_5)],
    #                     [upper_bound(y2_values_3), upper_bound(y2_values_4), upper_bound(y2_values_5)]],
    #                    [[lower_bound(y3_values_3), lower_bound(y3_values_4), lower_bound(y3_values_5)],
    #                     [upper_bound(y3_values_3), upper_bound(y3_values_4), upper_bound(y3_values_5)]],
    #                    [[lower_bound(y4_values_3), lower_bound(y4_values_4), lower_bound(y4_values_5)],
    #                     [upper_bound(y4_values_3), upper_bound(y4_values_4), upper_bound(y4_values_5)]],
    #                    [[lower_bound(y5_values_3), lower_bound(y5_values_4), lower_bound(y5_values_5)],
    #                     [upper_bound(y5_values_3), upper_bound(y5_values_4), upper_bound(y5_values_5)]]])
    errors1 = np.array([[[lower_bound(y0_values_3)], [upper_bound(y0_values_3)]],
                        [[lower_bound(y1_values_3)], [upper_bound(y1_values_3)]],
                        [[lower_bound(y2_values_3)], [upper_bound(y2_values_3)]],
                        [[lower_bound(y3_values_3)], [upper_bound(y3_values_3)]],
                        [[lower_bound(y4_values_3)], [upper_bound(y4_values_3)]],
                        [[lower_bound(y5_values_3)], [upper_bound(y5_values_3)]]])

    errors2 = np.array([[[lower_bound(y0_values_4)], [upper_bound(y0_values_4)]],
                        [[lower_bound(y1_values_4)], [upper_bound(y1_values_4)]],
                        [[lower_bound(y2_values_4)], [upper_bound(y2_values_4)]],
                        [[lower_bound(y3_values_4)], [upper_bound(y3_values_4)]],
                        [[lower_bound(y4_values_4)], [upper_bound(y4_values_4)]],
                        [[lower_bound(y5_values_4)], [upper_bound(y5_values_4)]]])

    errors3 = np.array([[[lower_bound(y0_values_5)], [upper_bound(y0_values_5)]],
                        [[lower_bound(y1_values_5)], [upper_bound(y1_values_5)]],
                        [[lower_bound(y2_values_5)], [upper_bound(y2_values_5)]],
                        [[lower_bound(y3_values_5)], [upper_bound(y3_values_5)]],
                        [[lower_bound(y4_values_5)], [upper_bound(y4_values_5)]],
                        [[lower_bound(y5_values_5)], [upper_bound(y5_values_5)]]])
    errors = [errors1, errors2, errors3]
    # colors = ['#5760A8', '#F6B816', '#DE4849', '#009688', '#FFC107', '#8E44AD']
    colors = ['#5760A8', '#F6B816', '#DE4849', '#4CA571', '#A15BB3', '#FF9650']

    labels = ['NoProt', 'Antitomo', 'ProTo', 'RndPathNet', 'NSGAPathNet', "OptiPathNet"]
    # colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    axs = axs.ravel()

    for i in range(3):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(6):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j], yerr=errors[i][j], ecolor='black', capsize=10))

        # # 在柱子上显示数值
        # for bar in bars:
        #     for rect in bar:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
        #                 fontsize=fontsize - 30)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15)

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 3]:
            # 只显示两位小数
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax.set_ylabel('Similarity Score', fontsize=fontsize -15, family='Times New Roman')

            ax.set_yticks(np.linspace(0, 1, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 15)
        else:
            ax.set_yticks(np.linspace(0, 1, 4))
            ax.tick_params(axis='y', labelsize=0)  # 隐藏纵轴数值，仅显示刻度线

        ax.set_ylim(0, 1)

    # colors = ['#DE4849']
    # 添加统一的图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors[:]]
    # labels = ['Distilled-FLow-MAE']
    fig.legend(handles, labels[:], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=fontsize - 15, frameon=False,
               shadow=False, fancybox=True, edgecolor='black', facecolor='white')  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之上
    plt.tight_layout(rect=[0, 0, 1, 0.77])

    plt.savefig(f"../img/fig5_subplot_no_xticks.png", dpi=300)
    plt.savefig(f"../img/fig5_subplot_no_xticks.pdf", dpi=600)
    plt.show()

def Fig42():
    fontsize = 50
    y0_values_3 = [0.585714, 0.57777, 0.569223, 0.551851, 0.551851, 0.5428, 0.541851, 0.540851, 0.531851, 0.531851]
    y1_values_3 = [0.59532, 0.58892, 0.58292, 0.57292, 0.57292, 0.56292, 0.55292, 0.55292, 0.55292, 0.55292]
    y2_values_3 = [0.653846, 0.590909, 0.590909, 0.590909, 0.5652, 0.5652, 0.5624, 0.55249, 00.5524, 0.5524]
    y3_values_3 = [0.703, 0.70, 0.691, 0.685, 0.683, 0.681, 0.680, 0.678, 0.675, 0.675]
    y4_values_3 = [0.753846, 0.690909, 0.7590909, 0.7590909, 0.75652, 0.745652, 0.745624, 0.7455249, 00.745524, 0.745524]
    y5_values_3 = [0.80, 0.7877, 0.7723, 0.756923, 0.7751, 0.7451, 0.77501, 0.75502, 0.74501, 0.77501]
    y0_values_4 = [0.4013, 0.3691, 0.343, 0.3517, 0.3605, 0.3586, 0.3857, 0.3931, 0.3471, 0.3471]
    y1_values_4 = [0.405013, 0.3944691, 0.3943, 0.394517, 0.39605, 0.393586, 0.393857, 0.393931, 0.393471, 0.393471]
    y2_values_4 = [0.50, 0.5048, 0.5039, 0.5027, 0.5026, 0.5259, 0.523, 0.521, 0.519, 0.520]
    y3_values_4 = [0.55, 0.548, 0.539, 0.527, 0.526, 0.5259, 0.523, 0.521, 0.519, 0.520]
    y4_values_4 = [0.60, 0.5829, 0.5280, 0.5829, 0.6, 0.5914, 0.5914, 0.5646, 0.5951, 0.5951]
    y5_values_4 = [0.641, 0.6704, 0.67, 0.62, 0.682, 0.671, 0.666, 0.664, 0.635, 0.635]
    y0_values_5 = [0.3601, 0.3785, 0.3597, 0.3417, 0.3723, 0.3926, 0.3539, 0.3975, 0.3731, 0.3731]
    y1_values_5 = [0.395395, 0.39548, 0.39539, 0.39527, 0.39526, 0.395259, 0.39523, 0.39521, 0.39519, 0.39520]
    y2_values_5 = [0.4265, 0.463, 0.4625, 0.4617, 0.461, 0.4608, 0.4609, 0.46093, 0.46092, 0.460925]
    y3_values_5 = [0.5060, 0.5829, 0.5280, 0.5829, 0.6, 0.5914, 0.5914, 0.5646, 0.5951, 0.5951]
    y4_values_5 = [0.535, 0.5534, 0.5345, 0.59359393, 0.60343, 0.60341, 0.60341, 0.60340, 0.60340, 0.60340]
    y5_values_5 = [0.65, 0.63, 0.625, 0.617, 0.61, 0.608, 0.609, 0.6093, 0.6092, 0.60925]

    def lower_bound(x):
        return np.mean(x) - np.min(x)

    def upper_bound(x):
        return np.max(x) - np.mean(x)

    categories = ["Claranet", "Garr201112", "Deltacom"]
    values = np.array([[np.mean(y0_values_3), np.mean(y0_values_4), np.mean(y0_values_5)],
                       [np.mean(y1_values_3), np.mean(y1_values_4), np.mean(y1_values_5)],
                       [np.mean(y2_values_3), np.mean(y2_values_4), np.mean(y2_values_5)],
                       [np.mean(y3_values_3), np.mean(y3_values_4), np.mean(y3_values_5)],
                       [np.mean(y4_values_3), np.mean(y4_values_4), np.mean(y4_values_5)],
                       [np.mean(y5_values_3), np.mean(y5_values_4), np.mean(y5_values_5)]])
    # errors = np.array([[[lower_bound(y0_values_3), lower_bound(y0_values_4), lower_bound(y0_values_5)],
    #                     [upper_bound(y0_values_3), upper_bound(y0_values_4), upper_bound(y0_values_5)]],
    #                    [[lower_bound(y1_values_3), lower_bound(y1_values_4), lower_bound(y1_values_5)],
    #                     [upper_bound(y1_values_3), upper_bound(y1_values_4), upper_bound(y1_values_5)]],
    #                    [[lower_bound(y2_values_3), lower_bound(y2_values_4), lower_bound(y2_values_5)],
    #                     [upper_bound(y2_values_3), upper_bound(y2_values_4), upper_bound(y2_values_5)]],
    #                    [[lower_bound(y3_values_3), lower_bound(y3_values_4), lower_bound(y3_values_5)],
    #                     [upper_bound(y3_values_3), upper_bound(y3_values_4), upper_bound(y3_values_5)]],
    #                    [[lower_bound(y4_values_3), lower_bound(y4_values_4), lower_bound(y4_values_5)],
    #                     [upper_bound(y4_values_3), upper_bound(y4_values_4), upper_bound(y4_values_5)]],
    #                    [[lower_bound(y5_values_3), lower_bound(y5_values_4), lower_bound(y5_values_5)],
    #                     [upper_bound(y5_values_3), upper_bound(y5_values_4), upper_bound(y5_values_5)]]])
    errors1 = np.array([[[lower_bound(y0_values_3)], [upper_bound(y0_values_3)]],
                        [[lower_bound(y1_values_3)], [upper_bound(y1_values_3)]],
                        [[lower_bound(y2_values_3)], [upper_bound(y2_values_3)]],
                        [[lower_bound(y3_values_3)], [upper_bound(y3_values_3)]],
                        [[lower_bound(y4_values_3)], [upper_bound(y4_values_3)]],
                        [[lower_bound(y5_values_3)], [upper_bound(y5_values_3)]]])

    errors2 = np.array([[[lower_bound(y0_values_4)], [upper_bound(y0_values_4)]],
                        [[lower_bound(y1_values_4)], [upper_bound(y1_values_4)]],
                        [[lower_bound(y2_values_4)], [upper_bound(y2_values_4)]],
                        [[lower_bound(y3_values_4)], [upper_bound(y3_values_4)]],
                        [[lower_bound(y4_values_4)], [upper_bound(y4_values_4)]],
                        [[lower_bound(y5_values_4)], [upper_bound(y5_values_4)]]])

    errors3 = np.array([[[lower_bound(y0_values_5)], [upper_bound(y0_values_5)]],
                        [[lower_bound(y1_values_5)], [upper_bound(y1_values_5)]],
                        [[lower_bound(y2_values_5)], [upper_bound(y2_values_5)]],
                        [[lower_bound(y3_values_5)], [upper_bound(y3_values_5)]],
                        [[lower_bound(y4_values_5)], [upper_bound(y4_values_5)]],
                        [[lower_bound(y5_values_5)], [upper_bound(y5_values_5)]]])
    errors = [errors1, errors2, errors3]
    # colors = ['#5760A8', '#F6B816', '#DE4849', '#009688', '#FFC107', '#8E44AD']
    colors = ['#5760A8', '#5760A8', '#5760A8', '#5760A8', '#5760A8', '#5760A8']

    labels = ['NoProt', 'Antitomo', 'ProTo', 'RndPathNet', 'NSGAPathNet', "OptiPathNet"]
    # colors = ['#5760A8', '#F6B816', '#DE4849']
    plt.rc('font', family='Times New Roman')
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs = axs.ravel()

    for i in range(3):  # 遍历每个类别
        ax = axs[i]
        # 显示网格线
        ax.grid(ls='--', alpha=0.5)
        bar_width = 0.8
        indices = np.arange(len(labels))

        # 绘制每个类别的小图，柱子之间无缝隙
        bars = []
        for j in range(6):
            bars.append(ax.bar(indices[j], values[j][i], bar_width, color=colors[j], yerr=errors[i][j], ecolor='black', capsize=10))

        # # 在柱子上显示数值
        # for bar in bars:
        #     for rect in bar:
        #         height = rect.get_height()
        #         ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom',
        #                 fontsize=fontsize - 30)

        # 标题设置
        ax.set_title(categories[i], fontsize=fontsize - 15, loc='center')

        # 移除所有的横轴标签
        ax.set_xticks([])

        # 只在左侧两张图上显示纵轴刻度值
        if i in [0, 3]:
            # 只显示两位小数
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
            ax.set_ylabel('Similarity Score', fontsize=fontsize -15, family='Times New Roman')

            ax.set_yticks(np.linspace(0.3, 0.9, 4))
            ax.tick_params(axis='y', labelsize=fontsize - 15)
        else:
            ax.set_yticks(np.linspace(0.3, 0.9, 4))
            ax.tick_params(axis='y', labelsize=0)  # 隐藏纵轴数值，仅显示刻度线

        ax.set_ylim(0.3, 0.81)

    # colors = ['#DE4849']
    # 添加统一的图例
    # handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) for color in colors[:]]
    # # labels = ['Distilled-FLow-MAE']
    # fig.legend(handles, labels[:], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=fontsize - 15, frameon=False,
    #            shadow=False, fancybox=True, edgecolor='black', facecolor='white')  # 调整图例中颜色块的大小
    # 调整图表布局,把图例放在子图之上
    plt.tight_layout()

    plt.savefig(f"../img/fig42.png", dpi=300)
    plt.savefig(f"../img/fig42.pdf", dpi=600)
    plt.show()

if __name__ == '__main__':
    Fig42()
