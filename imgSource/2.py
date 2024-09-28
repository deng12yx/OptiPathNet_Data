import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
"""
action=0代表小网络
action=1代表中等网络
action=2代表大网络
"""
action = 0
# 读取图文件
if action == 0:
    file_path = "../result/little_pre_ran.graphml"  # 攻击前的网路
    file_path2 = "../result/little_lat_ran.graphml"  # 攻击1后的网络
    file_path3 = "../result/l_lat.graphml"  # 攻击2后的网络
    file_path4 = "../result/gan_lat.graphml"    # GAN攻击的网络
elif action == 1:
    file_path = "../result/middle_pre_ran.graphml"
    file_path2 = "../result/middle_lat_ran.graphml"
    file_path3 = "../result/middle_lat.graphml"
    file_path4 = "../result/m_gan_lat.graphml"
else:
    file_path = "../result/large_pre_ran.graphml"
    file_path2 = "../result/large_lat_ran.graphml"
    file_path3 = "../result/lat.graphml"
    file_path4 = "../result/l_gan_lat.graphml"
G = nx.read_graphml(file_path)
G2 = nx.read_graphml(file_path2)
G3 = nx.read_graphml(file_path3)
G4 = nx.read_graphml(file_path4)
all_G = [G, G2, G3, G4]

origin_weights = []
for g in all_G:
    weight = {(u, v): "{:.2e}".format(d['weight']) for u, v, d in g.edges(data=True)}
    origin_weights.append(weight)

# 权重统一加1
for g in all_G:
    for (u, v, d) in g.edges(data=True):
        d['weight'] = float(d['weight']) + 1

weights = []
scaler = MinMaxScaler(feature_range=(0, 1))
for g in all_G:
    weight = np.array([float(d['weight']) for (u, v, d) in g.edges(data=True)])
    weight = scaler.fit_transform(weight.reshape(-1, 1)).flatten()
    weights.append(weight)

cmap = plt.cm.get_cmap('coolwarm')

# 绘制图
# pos = nx.spring_layout(all_G[0], seed=42, iterations=100, threshold=1e-8, center=[0.5, 0.5], scale=1, dim=2)
pos = nx.spring_layout(all_G[0], seed=42, iterations=50)
count = 33
# 主图中的节点和边
primary_nodes = set(all_G[0].nodes())
primary_edges = set(all_G[0].edges())
for g, weight, origin_weight in zip(all_G, weights, origin_weights):
    # 判断是否所有节点都在pos中，如果不在就对没有位置的节点进行随机分布
    for node in g.nodes():
        if node not in pos:
            pos[node] = np.random.rand(2)

    # 设置颜色和宽度
    colors = [cmap(w) for w in weight]
    widths = [w * 10 for w in weight]  # 调整宽度比例

    # 设置节点大小和颜色
    degrees = dict(g.degree())
    node_sizes = [degrees[node] * 150 for node in g.nodes()]
    node_colors = [degrees[node] for node in g.nodes()]
    # 绘制图
    fig = plt.figure(figsize=(10, 10))  # 调整图形大小

    # 分别绘制主图中的节点和边，以及其余图中的节点和边
    primary_nodes_in_g = [node for node in g.nodes() if node in primary_nodes]
    secondary_nodes_in_g = [node for node in g.nodes() if node not in primary_nodes]
    primary_edges_in_g = [edge for edge in g.edges() if edge in primary_edges]
    secondary_edges_in_g = [edge for edge in g.edges() if edge not in primary_edges]

    # 分别生成节点大小数组
    primary_node_sizes = [node_sizes[list(g.nodes()).index(node)] for node in primary_nodes_in_g]
    secondary_node_sizes = [node_sizes[list(g.nodes()).index(node)] for node in secondary_nodes_in_g]
    primary_color = [colors[list(g.edges()).index(edge)] for edge in primary_edges_in_g]
    secondary_color = [colors[list(g.edges()).index(edge)] for edge in secondary_edges_in_g]
    primary_width = [widths[list(g.edges()).index(edge)] for edge in primary_edges_in_g]
    secondary_width = [widths[list(g.edges()).index(edge)] for edge in secondary_edges_in_g]
    # 绘制图
    plt.figure(figsize=(12, 12))  # 调整图形大小

    # 主图中的节点
    nx.draw_networkx_nodes(g, pos, nodelist=primary_nodes_in_g, node_size=primary_node_sizes, node_color="CornflowerBlue", alpha=0.6,
                           linewidths=1, edgecolors='black')
    # 其余图中的节点（虚线）
    nx.draw_networkx_nodes(g, pos, nodelist=secondary_nodes_in_g, node_size=secondary_node_sizes, node_color="CornflowerBlue", alpha=0.6,
                           linewidths=1, edgecolors='black', node_shape='s')  # 使用方形节点

    # 主图中的边
    nx.draw_networkx_edges(g, pos, edgelist=primary_edges_in_g, edge_color=primary_color, width=primary_width)
    # 其余图中的边（虚线）
    nx.draw_networkx_edges(g, pos, edgelist=secondary_edges_in_g, edge_color=secondary_color, width=secondary_width, style='dashed')

    plt.axis('off')  # 关闭坐标轴

    # # 添加图例
    # from matplotlib.lines import Line2D
    #
    # legend_labels = {"High Degree": "red", "Low Degree": "blue"}
    # legend_handles = [Line2D([0], [0], marker='o', color='w', label=label,
    #                          markerfacecolor=color, markersize=20) for label, color in legend_labels.items()]
    # plt.legend(handles=legend_handles, frameon=False, loc='upper right', title='Node Degree', fontsize=25,
    #            title_fontsize=30, bbox_to_anchor=(1, 1), edgecolor='black', facecolor='white', shadow=False,
    #            fancybox=True)
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(weight), vmax=max(weight)))
    sm.set_array([])
    # 添加颜色条
    cbaxes = plt.gcf().add_axes([0.85, 0.65, 0.03, 0.2])  # 调整颜色条的位置和大小
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(weight), vmax=max(weight)))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbaxes)  # 调整颜色条的大小
    cbar.set_label('Edge Weight', fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    # 保存图形
    plt.savefig(f"../img/fig{count}.png", dpi=600, format='png', bbox_inches='tight')
    plt.savefig(f"../img/fig{count}.pdf", dpi=600, format='pdf', bbox_inches='tight')

    # 显示图形
    plt.show()
    plt.close()
    count += 1
