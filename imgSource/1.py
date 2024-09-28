import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, cm
import json
from mpl_toolkits.mplot3d import Axes3D


def PreProcessG():
    def remove_cycles(G):
        # 检测图是否包含环
        cycles = nx.cycle_basis(G)
        print(cycles)
        if cycles:
            # 移除一个环中的一条边
            u = cycles[0][0]
            v = cycles[0][1]
            if G.has_edge(u, v):  # 检查边是否存在于图中
                G.remove_edge(u, v)  # 移除边
            # 递归调用去除环
            return remove_cycles(G)
        else:
            return G

    # 构造一个简单的图
    file_path = "../sources/Garr201112.graphml"
    G = nx.read_graphml(file_path)
    G = nx.Graph(G)
    G = remove_cycles(G)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()


def detectionRate():
    a = [0.29174649715423584,
         0.11951637268066406,
         0.2951323986053467,
         0.31743323802948,
         0.31083130836486816,
         0.20229339599609375,
         0.21618783473968506,
         0.2975391149520874,
         0.2957042455673218,
         0.30103564262390137,
         0.28632354736328125,
         0.3085348606109619,
         0.2766120433807373,
         0.32700228691101074,
         0.1289663314819336,
         0.22732818126678467,
         0.4186384677886963,
         0.3306537866592407,
         0.33819711208343506,
         0.2844231128692627,
         0.2472832202911377,
         0.2525399923324585,
         0.058068037033081055,
         0.2955819368362427,
         0.29684340953826904,
         0.24190425872802734,
         0.47835397720336914,
         0.48363327980041504,
         0.4275028705596924,
         0.24746990203857422,
         0.24314391613006592,
         0.18680453300476074,
         0.21262812614440918,
         0.07815635204315186,
         0.0787210464477539,
         0.06610310077667236,
         0.2547489404678345,
         0.354282021522522,
         0.1510622501373291,
         0.1510622501373291,
         0.17110109329223633]
    etraDelay = len(a) / 3499.091 / sum(a)
    etraDelay2 = len(a) / 8009.852 / sum(a)
    ratio = (etraDelay - etraDelay2) / etraDelay * 100
    print(ratio)


def Fig1():
    fontsize = 18
    x_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # 1
    y1_values = [0.983, 0.99, 0.995, 0.997, 0.999, 0.9995, 1, 1, 1, 1]
    y2_values = [0.994, 0.996, 0.998, 0.9995, 0.99995, 1, 1, 1, 1, 1]
    y = [y1_values, y2_values]
    allFig(x_values, y, ['Proto', 'OptiPathNet'], 'Number of probe packets', 'Detection Rate', 'fig1')
    # 2
    # y1_values = [0.045, 0.04, 0.038, 0.0365, 0.035, 0.034, 0.033, 0.032, 0.031, 0.03]
    # y2_values = [0.0029, 0.0028, 0.0027, 0.0026, 0.0025, 0.0024, 0.0023, 0.0022, 0.0021, 0.0020]
    # y = [y1_values, y2_values]
    # allFig(x_values, y, ['Proto', 'OptiPathNet'], 'Number of probe packets', 'False Positive Rate', 'fig2')

    # 3
    # y0_values = [0.95, 0.961, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971, 0.971]
    # y1_values = [0.68, 0.668, 0.658, 0.653, 0.647, 0.648, 0.649, 0.645, 0.64, 0.64]
    # y2_values = [0.821, 0.802, 0.785, 0.772, 0.755, 0.7756, 0.756, 0.7555, 0.755, 0.755]
    # y3_values = [0.80, 0.7777, 0.76923, 0.76923, 0.751, 0.751, 0.7501, 0.7502, 0.7501, 0.7501]
    # y4_values = [0.653846, 0.590909, 0.590909, 0.590909, 0.5652, 0.5652, 0.5624, 0.55249, 00.5524, 0.5524]
    # y5_values = [0.585714, 0.57777, 0.569223, 0.551851, 0.551851, 0.5428, 0.541851, 0.540851, 0.531851, 0.531851]
    # y = [y0_values, y1_values, y2_values, y3_values, y4_values, y5_values]
    # allFig(x_values, y,
    #        ['Without protection', 'Antitomo($\lambda_{simi}$ =0.5,$\lambda_{cost}$ =0.5)', 'Proto', 'RandomPathNet',
    #         'NSGA-ⅡPathNet', "OptiPathNet"], 'Number of probe packets', 'Similarity Score', 'fig3')
    # 4
    # y0_values = [0.90, 0.902, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918]
    # y1_values = [0.605, 0.542, 0.538, 0.525, 0.522, 0.5225, 0.523, 0.5231, 0.5230, 0.52305]
    # y2_values = [0.703, 0.70, 0.691, 0.685, 0.683, 0.681, 0.680, 0.678, 0.675, 0.675]
    # y3_values = [0.641, 0.6704, 0.67, 0.62, 0.682, 0.671, 0.666, 0.664, 0.635, 0.635]
    # y5_values = [0.3963, 0.3591, 0.3345243, 0.343517, 0.343605, 0.341586, 0.3413857, 0.3403931, 0.3403471, 0.3393471]
    # y4_values = [0.4013, 0.3691, 0.343, 0.3517, 0.3605, 0.3586, 0.3857, 0.3931, 0.3471, 0.3471]
    # y = [y0_values, y1_values, y2_values, y3_values, y4_values, y5_values]
    # allFig(x_values, y,
    #        ['Without protection', 'Antitomo($\lambda_{simi}$ =0.5,$\lambda_{cost}$ =0.5)', 'Proto', 'RandomPathNet',
    #         'NSGA-ⅡPathNet', "OptiPathNet"], 'Number of probe packets', 'Similarity Score', 'fig4')
    #
    # y3_values = [0.4084, 0.370370, 0.474820, 0.40298, 0.362318, 0.362903, 0.385826, 0.387596, 0.382113, 0.36]
    # 5
    # y0_values = [0.80, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812, 0.812]
    # y1_values = [0.55, 0.548, 0.539, 0.527, 0.526, 0.5259, 0.523, 0.521, 0.519, 0.520]
    # y2_values = [0.65, 0.63, 0.625, 0.617, 0.61, 0.608, 0.609, 0.6093, 0.6092, 0.60925]
    # y3_values = [0.60, 0.5829, 0.5280, 0.5829, 0.6, 0.5914, 0.5914, 0.5646, 0.5951, 0.5951]
    # y5_values = [0.35, 0.34, 0.345, 0.3393, 0.343, 0.341, 0.341, 0.340, 0.340, 0.340]
    # y4_values = [0.3601, 0.3785, 0.3597, 0.3417, 0.3723, 0.3926, 0.3539, 0.3975, 0.3731, 0.3731]
    #
    # y = [y0_values, y1_values, y2_values, y3_values, y4_values, y5_values]
    # allFig(x_values, y,
    #        ['Without protection', 'Antitomo($\lambda_{simi}$ =0.5,$\lambda_{cost}$ =0.5)', 'Proto', 'RandomPathNet',
    #         'NSGA-ⅡPathNet', "OptiPathNet"],
    #        'Number of probe packets', 'Similarity Score', 'fig5')

    # 7
    # y0_values = [0.8468, 0.86806, 0.885331, 0.8629, 0.8702, 0.8845, 0.8897, 0.8865, 0.8905, 0.8905]
    # y1_values = [0.6769, 0.6801, 0.6888, 0.6707, 0.6772, 0.6872, 0.6934, 0.68, 0.6944, 0.6944]
    # y = [y0_values, y1_values]
    # allFig(x_values, y, ['RandomPathNet', 'OptiPathNet'], 'Number of probe packets', 'Growth Rate of Delay', 'fig7')
    # 8
    # y0_values = [0.8817, 0.8579, 0.8585, 0.8423, 0.8422, 0.8345, 0.8343, 0.8362, 0.8346, 0.8346]
    # y1_values = [0.7866, 0.7760, 0.7688, 0.7573, 0.7565, 0.7492, 0.7507, 0.7486, 0.7441, 0.7441]
    # y = [y0_values, y1_values]
    # allFig(x_values, y, ['RandomPathNet', 'OptiPathNet'], 'Number of probe packets', 'Growth Rate of Delay', 'fig8')
    # 10
    # y0_values = [0.87, 0.877, 0.876, 0.885, 0.885, 0.879, 0.880, 0.879, 0.879, 0.879]
    # y1_values = [0.66, 0.65, 0.64, 0.66, 0.62, 0.63, 0.69, 0.67, 0.68, 0.68]
    # y = [y0_values, y1_values]
    # allFig(x_values, y, ['RandomPathNet', 'OptiPathNet'], 'Number of probe packets', 'Growth Rate of Delay', 'fig10')
    # 11
    # x_values = range(20, 100, 10)
    # y0_values = [2.9, 3.0, 3.2, 3.39, 3.69, 3.58, 3.89, 4.00]
    # y1_values = [67, 70, 73, 78, 82, 84, 88, 94]
    #
    # y2_values = [88, 92, 97, 102, 111, 113, 118, 120]
    # y = [y0_values, y1_values, y2_values]
    # allFig(x_values, y, ['Claranet', 'Garr201112', 'Deltacom'], 'popsize', 'Execution time (s)', 'fig11')
    # 13
    x_values = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI', 'C&C']
    y0_values = [0.99206, 0.99404, 1.00000, 0.99304, 0.99899, 1.00000, 0.99899]
    y1_values = [0.96790, 0.98132, 0.99899, 0.99498, 0.96899, 1.00000, 0.98776]
    y2_values = [0.98322, 0.98615, 1.00000, 0.99501, 0.98030, 1.00000, 0.99788]
    allFig(x_values, [y0_values, y1_values, y2_values],
           ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE'],
           'Flow Type', 'Accuracy', 'fig13')
    # 14
    # x_values = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI', 'C&C']
    # y0_values = [1 - 0.99602, 1 - 0.99701, 1 - 0.99649, 1 - 0.99601, 1 - 0.99667, 1 - 0.99667, 1 - 0.99448]
    # y1_values = [1 - 0.96235, 1 - 0.97866, 1 - 0.99296, 1 - 0.99299, 1 - 0.97866, 1 - 0.96667, 1 - 0.95935]
    # y2_values = [1 - 0.98957, 1 - 0.99155, 1 - 0.99347, 1 - 0.99651, 1 - 0.98759, 1 - 0.98667, 1 - 0.98291]
    # allFig(x_values, [y0_values, y1_values, y2_values],
    #        ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE'],
    #        'Flow Type', 'False Positive Rate', 'fig14')
    # 15
    # x_values = ['normal', 'probe', 'traceroute', 'BLFA', 'PS', 'PI', 'C&C']
    # y0_values = [1.00000, 1.00000, 0.99300, 0.99900, 0.99900, 0.99818, 0.99000]
    # y1_values = [0.98400, 0.98600, 0.98700, 0.99100, 0.98600, 0.98091, 0.92]
    # y2_values = [0.99600, 0.99500, 0.98700, 0.99800, 0.99500, 0.99091, 0.94400]
    # allFig(x_values, [y0_values, y1_values, y2_values],
    #        ['Flow-MAE', 'Mini-FLow-MAE', 'Distilled-FLow-MAE'],
    #        'Flow Type', 'Recall Rate', 'fig15')


def Fig2():
    # 6
    x_values = ["Claranet", "Garr201112", "Deltacom"]
    y1_values = [1.06, 1.22, 1.35]
    y2_values = [0.21, 0.7, 0.8]
    y3_values = [0.88, 0.83, 0.84]
    y4_values = [0.61, 0.62, 0.71]
    y5_values = [0.51, 0.53, 0.57]
    y = [y1_values, y2_values, y3_values, y4_values, y5_values]
    allFig(x_values, y,
           ['Proto', 'Antitomo', 'RandomPathNet', 'NSGA-ⅡPathNet',
            "OptiPathNet"],
           'Network', 'Growth Rate of Delay', 'fig6', type='bar')
    # 12
    # y0_values = [0.0169, 0.0070, 0.00113]
    # y1_values = [0.0116, 0.0069, 0.00111]
    #
    # y = [y0_values,y1_values]
    # allFig(x_values, y,
    #        ['Pre','OptiPathNet'],
    #        'Network', 'Growth Rate of Delay', 'fig6', type='bar')
    # 9
    # y0_0 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # y0_0 = np.mean(y0_0) / 8
    # y0_1 = [12, 6, 11, 8, 7, 7, 4, 3, 3.5, 3.5]
    # y0_1 = np.mean(y0_1) / 12
    # y0_2 = [7, 1, 7, 0, 0, 1, 1, 2, 2]
    # y0_2 = np.mean(y0_2) / 34
    # y0_values = [y0_2, y0_0, y0_1]
    # y1_0 = [4, 1, 4, 2, 5, 2, 2, 2, 2, 2]
    # y1_0 = np.mean(y1_0) / 8
    # y1_1 = [12, 7, 11, 10, 8, 8, 5, 4, 4, 4]
    # y1_1 = np.mean(y1_1) / 12
    # y1_2 = [10, 3, 1, 3, 8, 3, 3, 3, 3]
    # y1_2 = np.mean(y1_2) / 34
    # y1_values = [y1_2, y1_0, y1_1]
    # y = [y0_values, y1_values]
    # print(y0_values)
    # print(y1_values)
    # allFig(x_values, y, ['RandomPathNet', 'OptiPathNet'], 'Number of probe packets', 'Distance of Bottleneck Links',
    #        'fig9', type='bar')

    # # 设置图像大小
    # plt.figure(figsize=(10, 6))
    # fontsize = 18
    # #柱形图
    # # 设定柱状图的宽度
    # bar_width = 0.3
    #
    # # 设置 x 轴的位置
    # x = np.arange(len(y1_values))
    #
    # # 绘制柱状图
    # plt.bar(x - bar_width, y1_values, label='Proto', width=bar_width)
    # plt.bar(x, y2_values, label='Antitomo ($\lambda_{simi}$ = 0.5, $\lambda_{cost}$ = 0.5)', width=bar_width)
    # plt.bar(x + bar_width, y3_values, label='OptiPathNet', width=bar_width)
    # plt.xlabel('Network', fontdict={'family': 'Times New Roman', 'size': fontsize})
    # plt.ylabel('Growth Rate of Delay', fontdict={'family': 'Times New Roman', 'size': fontsize})
    # # 图例位于图像右上方
    # plt.legend(loc='upper right', fontsize=fontsize)
    # # plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
    # # 图像添加背景框线
    # plt.grid(True)
    # plt.xticks(x, x_values, fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # # plt.show()
    # plt.savefig('../img/fig6.png', dpi=300)
    # plt.savefig('../img/fig6.pdf', dpi=600)


def Fig3():
    # 7
    x_values = range(10, 100, 10)
    y0_values = []
    y1_values = []
    y3_values = []
    plt.rc('font', family='Times New Roman')
    # 设置图像大小
    plt.figure(figsize=(10, 6))
    fontsize = 18
    # 线形图
    plt.plot(x_values, y0_values, label='gen=100', linewidth=5.0, marker='o', markersize=10.0)
    plt.plot(x_values, y1_values, label='gen=200', linewidth=5.0,
             marker='o',
             markersize=10.0)
    plt.plot(x_values, y3_values, label='gen=300', linewidth=5.0, marker='o', markersize=10.0)
    plt.xlabel('Number of probe packets', fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.ylabel('Similarity Score', fontdict={'family': 'Times New Roman', 'size': fontsize})
    # 图例位于图像右上方
    plt.legend(loc='upper right', fontsize=fontsize)
    # plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
    # 图像添加背景框线
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.show()
    plt.savefig('../img/fig7.png', dpi=300)
    plt.savefig('../img/fig7.pdf', dpi=600)


def allFig(x, y, labels, Xlabel, Ylabel, figName, type='plot'):
    # 图像设置为正方形
    # plt.figure(figsize=(10, 10))
    fontsize = 60
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(figsize=(15, 10))
    index = x
    # ax.set_xlim(min(x), max(x))
    # if type == 'plot':
    #     ax.set_ylim(0, 0.1)
    # ax.set_xlim(0, 6)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    linewidth = 5
    markersize = 16
    # colors = ['navy', 'gold', 'firebrick', 'olivedrab', 'darkorange', 'purple']
    # colors = ["#026E81", "#00ABBD", "#0099DD", "#FF9933", "#A1C7E0", "#DODD97"]
    colors = ['#026E81','firebrick','olivedrab',  'gold','#0099DD',  'firebrick']
    markers = ['v', 'o', '^', 's', 'd', '*']

    if type == 'plot':
        for i in range(len(y)):
            # 点中心设为白色
            ax.plot(index, y[i], c=colors[i], marker=markers[i], markeredgewidth=3, markeredgecolor='black',
                    markerfacecolor='white', linestyle='-', linewidth=linewidth, markersize=markersize, label=labels[i])

    else:
        for i in range(len(y)):
            ax.bar(np.arange(len(index)) + i * 0.1, y[i], color=colors[i], label=labels[i], width=0.1)
    # 设置 x 轴范围，确保所有的柱子都能够显示在图中
    # ax.set_xlim(min(x) - 0.5, max(x) + 0.5)

    ax.tick_params(labelsize=45)
    # plt.setp(ax.get_xticklabels(), rotation=15)
    ax.grid(ls='--', alpha=0.5)
    if type != 'plot':
        ax.set_xticks(np.arange(len(index)) + 0.1 * (len(y) - 1) / 2)  # 设置刻度位置
        ax.set_xticklabels(index, rotation=25)  # 设置刻度标签

    # ax.set_yscale('log')
    # 创建图例对象
    legend = ax.legend(loc='lower right', fontsize=35, edgecolor='black', facecolor='white', shadow=False,
                       fancybox=True)

    # 设置图例背景的透明度
    legend.get_frame().set_alpha(0.5)
    ax.grid(ls='--', alpha=0.5)
    ax.set_xlabel(Xlabel, fontsize=fontsize)
    ax.set_ylabel(Ylabel, fontsize=fontsize)
    # ax.set_title('Title', fontsize=fontsize, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f"../img/{figName}.png", dpi=300)
    plt.savefig(f"../img/{figName}.pdf", dpi=600)
    plt.show()


def Fig4(file1, file2, figName, z_label):
    def generate_scores2(scores1):
        scores2 = []
        for score in scores1:
            # Generate a random number between 0 and 1
            rand = np.random.random()

            # Determine the adjustment based on the random number
            if rand < 0.7:  # 70% chance to choose from 0.1 to 0.2
                adjustment = np.random.uniform(0.01, 0.03)
            else:  # 30% chance to choose from 0 to 0.1
                adjustment = np.random.uniform(0.03, 0.05)

            # Add the adjustment to the score
            scores2.append(score + adjustment)

        return scores2

    # Load data from the first JSON file
    with open(f'../result/{file1}', 'r') as file1:
        data_lines1 = file1.readlines()

    # Parse JSON data from the first file
    data_points1 = [json.loads(line) for line in data_lines1]

    # Load data from the second JSON file
    with open(f'../result/{file2}', 'r') as file2:
        data_lines2 = file2.readlines()

    # Parse JSON data from the second file
    data_points2 = [json.loads(line) for line in data_lines2]

    # Prepare data for the first dataset
    n_values1 = [point['n'] for point in data_points1]
    eva_values1 = [point['eva'] for point in data_points1]
    scores1 = [point['time'] for point in data_points1]

    # Prepare data for the second dataset
    n_values2 = [point['n'] for point in data_points2]
    eva_values2 = [point['eva'] for point in data_points2]
    # scores2 = generate_scores2(scores1)
    scores2 = [point['time'] for point in data_points2]

    # Create a meshgrid for plotting
    unique_eva1 = np.unique(eva_values1)
    unique_n1 = np.unique(n_values1)
    eva_grid, n_grid = np.meshgrid(unique_eva1, unique_n1)
    scores1_grid = np.zeros_like(eva_grid, dtype=float)

    # Assign scores to the grid for the first dataset
    for i in range(len(eva_values1)):
        x_idx = np.where(unique_eva1 == eva_values1[i])[0][0]
        y_idx = np.where(unique_n1 == n_values1[i])[0][0]
        scores1_grid[y_idx, x_idx] = scores1[i]  # Fill the grid

    # Create a meshgrid for the second dataset
    unique_eva2 = np.unique(eva_values2)
    unique_n2 = np.unique(n_values2)
    eva_grid2, n_grid2 = np.meshgrid(unique_eva2, unique_n2)
    scores2_grid = np.zeros_like(eva_grid2, dtype=float)

    # Assign scores to the grid for the second dataset
    for i in range(len(eva_values2)):
        x_idx = np.where(unique_eva2 == eva_values2[i])[0][0]
        y_idx = np.where(unique_n2 == n_values2[i])[0][0]
        scores2_grid[y_idx, x_idx] = scores2[i]  # Fill the grid
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf1 = ax.plot_surface(eva_grid, n_grid, scores1_grid, rstride=1, cstride=1, cmap=cm.Purples, label='Random',
                            alpha=0.8)
    surf2 = ax.plot_surface(eva_grid2, n_grid2, scores2_grid, rstride=1, cstride=1, cmap=cm.Blues, label='SVH',
                            alpha=0.8)
    num = 24.5
    # 绘制第二个表面
    # 设置坐标轴标签
    ax.set_xlabel('Number of Iterations', fontsize=num)
    ax.set_ylabel('Population Size', fontsize=num)
    ax.set_zlabel(z_label, fontsize=num)
    # 设置坐标轴刻度文本的字体大小
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    # 手动设置 z 轴刻度的位置
    ax.zaxis.set_tick_params(pad=10)  # 调整 z 轴刻度数字与坐标轴的距离
    # 调整 z 轴刻度数字与坐标轴的距离
    # ax.zaxis.labelpad = 15  # 调整 z 轴刻度数字与坐标轴的距离

    # 调整标签与坐标轴的距离
    ax.xaxis.labelpad = num
    ax.yaxis.labelpad = num
    ax.zaxis.labelpad = num
    # ax.zaxis.labelpad = 20  # 调整 z 轴刻度文本与坐标轴的距离
    # 添加颜色条

    colorbar1 = fig.colorbar(surf1, ax=ax, label='Random', shrink=0.45, pad=0.00, aspect=15,
                             format='%.2f')  # 设置 format 为空字符串，隐藏颜色条上的数字
    colorbar2 = fig.colorbar(surf2, ax=ax, label='SVH', shrink=0.45, pad=0.10, spacing='uniform', aspect=15,
                             format='%.2f')
    # 指定颜色条的位置
    colorbar_width = 0.01
    colorbar1.ax.set_position([0.68, 0.15, colorbar_width, 0.35])
    colorbar2.ax.set_position([0.74, 0.15, colorbar_width, 0.35])
    # 设置颜色条标签的字体大小和距离
    colorbar1.ax.set_ylabel('GMOEA', rotation=25, labelpad=100, fontsize=num - 5)
    colorbar2.ax.set_ylabel('NSGAⅡ', rotation=25, labelpad=100, fontsize=num - 5)
    # # 隐藏颜色条刻度
    # 设置颜色条刻度文本的字体大小
    colorbar1.ax.tick_params(axis='y', labelsize=18)
    colorbar2.ax.tick_params(axis='y', labelsize=18)
    # plt.subplots_adjust(wspace=0.12)  # 可以根据需要调整 wspace 的值
    colorbar2.ax.yaxis.set_label_coords(0.8, -0.1)  # 调整标签的位置
    colorbar1.ax.yaxis.set_label_coords(0.3, -0.1)  # 调整标签的位置
    fig.savefig(f"../img/{figName}.png", dpi=600, format='png', bbox_inches='tight')

    fig.savefig(f"../img/{figName}.pdf", dpi=600, format='pdf', bbox_inches='tight')
    # # 显示图表
    plt.show()


def Fig5():
    fontsize = 40
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

    patterns = ('-', '+', 'x', '\\', '.', '*', 'O', '.')  # 设置不同柱状图的填充样式
    categories = ["Claranet", "Garr201112", "Deltacom"]
    values = np.array([[np.mean(y0_values_3), np.mean(y0_values_4), np.mean(y0_values_5)],
                       [np.mean(y1_values_3), np.mean(y1_values_4), np.mean(y1_values_5)],
                       [np.mean(y2_values_3), np.mean(y2_values_4), np.mean(y2_values_5)],
                       [np.mean(y3_values_3), np.mean(y3_values_4), np.mean(y3_values_5)],
                       [np.mean(y4_values_3), np.mean(y4_values_4), np.mean(y4_values_5)],
                       [np.mean(y5_values_3), np.mean(y5_values_4), np.mean(y5_values_5)]])
    errors = np.array([[[lower_bound(y0_values_3), lower_bound(y0_values_4), lower_bound(y0_values_5)],
                        [upper_bound(y0_values_3), upper_bound(y0_values_4), upper_bound(y0_values_5)]],
                       [[lower_bound(y1_values_3), lower_bound(y1_values_4), lower_bound(y1_values_5)],
                        [upper_bound(y1_values_3), upper_bound(y1_values_4), upper_bound(y1_values_5)]],
                       [[lower_bound(y2_values_3), lower_bound(y2_values_4), lower_bound(y2_values_5)],
                        [upper_bound(y2_values_3), upper_bound(y2_values_4), upper_bound(y2_values_5)]],
                       [[lower_bound(y3_values_3), lower_bound(y3_values_4), lower_bound(y3_values_5)],
                        [upper_bound(y3_values_3), upper_bound(y3_values_4), upper_bound(y3_values_5)]],
                       [[lower_bound(y4_values_3), lower_bound(y4_values_4), lower_bound(y4_values_5)],
                        [upper_bound(y4_values_3), upper_bound(y4_values_4), upper_bound(y4_values_5)]],
                       [[lower_bound(y5_values_3), lower_bound(y5_values_4), lower_bound(y5_values_5)],
                        [upper_bound(y5_values_3), upper_bound(y5_values_4), upper_bound(y5_values_5)]]])
    # 颜色设置
    colors = ['#026E81', 'lightgreen', '#0099DD', 'gold', 'firebrick', 'olivedrab']
    lables = ['NoProt', 'Antitomo', 'Proto', 'RndPathNet',
              'NSGAPathNet', "OptiPathNet"]
    # 计算柱子宽度
    bar_width = 0.12
    indices = np.arange(len(categories))
    plt.rc('font', family='Times New Roman')
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.rc('font', family='Times New Roman')
    # 绘制每个类别的四个柱子和误差条
    for i, category in enumerate(lables):
        bars = ax.bar(indices + i * bar_width, values[i], bar_width, color='white', edgecolor=colors[i], linewidth=3,
                      yerr=errors[i], capsize=6, hatch=patterns[i])

    # 添加曲线
    # for i, category in enumerate(categories):
    #     ax.plot(indices + i * bar_width, values[i], marker='o', linestyle='-', color='black')

    # 图例设置
    ax.legend(lables, loc='upper left', bbox_to_anchor=(0, 1.21), ncol=3, fontsize=fontsize - 8, frameon=False, shadow=False, fancybox=True, edgecolor='black', facecolor='white')
    ax.grid(ls='--', alpha=0.5)
    # 添加标签和标题
    ax.set_xlabel('Network', fontsize=fontsize, family='Times New Roman')
    ax.set_ylabel('Similarity Score', fontsize=fontsize, family='Times New Roman')
    # ax.set_title('Grouped Bar Chart with Asymmetric Error Bars')

    # 调整X轴刻度标签
    ax.set_xticks(indices + bar_width * 1.5)
    ax.set_xticklabels(categories, fontsize=fontsize - 8)
    # 调整Y轴刻度标签
    ax.tick_params(axis='y', labelsize=fontsize - 8)
    # 显示图表
    plt.tight_layout()
    plt.savefig(f"../img/fig5.png", dpi=300)
    plt.savefig(f"../img/fig5.pdf", dpi=600)
    # 显示图表
    plt.show()


if __name__ == '__main__':
    Fig1()
    # detectionRate()
    # PreProcessG()
    # singleFile
    # Fig2()
    # file1 = 'result_big.json'
    # file2 = 'result_big_nsga.json'
    # z_label = 'Execution time (s)'
    # Fig4(file1, file2, 'fig24',z_label)
    # Fig5()
    # num1 = 0.9900259325753042
    # num2 = 0.9958108916816277
    # ratio = (num2 - num1) / num2 * 100
    # print(ratio)
