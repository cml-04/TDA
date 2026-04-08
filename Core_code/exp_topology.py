"""
exp_topology.py — Mapper 拓扑结构与稳定性分析
=============================================
来源: main.py (TrajectoryAnalyzer, plot_mapper_graph, stability_sweep)
章节: 4.2 补充 (拓扑图), 4.5 参数稳定性
输出:
  fig/mapper_topology.png     Mapper 拓扑图 (节点角色着色)
  fig/gene_dynamics.png       沿拓扑路径的基因表达动态
  fig/parameter_stability.png 参数稳定性扫描结果
  fig/topology.html           Mapper 交互式可视化
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from scipy.sparse import issparse
from sklearn.cluster import DBSCAN
import kmapper as km

from core import (RANDOM_SEED, fig_path, make_fig, full_pipeline, auto_eps, ensure_fig_dir)

ensure_fig_dir('topology')
figp = make_fig('topology')


# ============================================================
# 节点角色分类
# ============================================================
def classify_nodes(graph, G):
    """Terminal / Bridge / Hub / Chain / Isolated"""
    artic = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if len(sub) > 2:
            artic.update(nx.articulation_points(sub))
    records = []
    for n in G.nodes():
        d = G.degree(n)
        s = len(graph['nodes'][n])
        if d == 0: r = 'Isolated'
        elif d == 1: r = 'Terminal'
        elif n in artic: r = 'Bridge'
        elif d >= 3: r = 'Hub'
        else: r = 'Chain'
        records.append({'node_id': n, 'degree': d, 'n_cells': s, 'role': r})
    df = pd.DataFrame(records)
    for role in ['Terminal', 'Bridge', 'Hub', 'Chain', 'Isolated']:
        cnt = (df['role'] == role).sum()
        if cnt > 0:
            print(f"  [{role}] {cnt}")
    return df


# ============================================================
# Mapper 图可视化
# ============================================================
def plot_mapper_graph(graph, G, node_df, highlight=None, title="Mapper", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=1.5, iterations=100)
    rc = {'Terminal': '#e74c3c', 'Bridge': '#f39c12', 'Hub': '#2ecc71',
          'Chain': '#95a5a6', 'Isolated': '#8e44ad'}
    colors = [rc.get(node_df[node_df['node_id'] == n]['role'].values[0], '#95a5a6')
              for n in G.nodes()]
    sizes = [len(graph['nodes'][n]) * 5 + 30 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                            alpha=0.85, edgecolors='white', linewidths=0.5)
    if highlight:
        hp = {n: pos[n] for n in highlight if n in pos}
        if hp:
            nx.draw_networkx_nodes(G, hp, nodelist=list(hp.keys()),
                node_color='none',
                node_size=[sizes[list(G.nodes()).index(n)] + 60 for n in hp],
                edgecolors='red', linewidths=3, ax=ax)
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w',
              markerfacecolor=c, markersize=10, label=r) for r, c in rc.items()],
              loc='upper left')
    ax.set_title(title, fontsize=14); ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================
# 轨迹分析
# ============================================================
def identify_root(G):
    largest = max(nx.connected_components(G), key=len)
    root = max(largest, key=lambda n: G.degree(n))
    print(f"[Info] Root: {root} (deg={G.degree(root)}, comp={len(largest)} nodes)")
    return root


def find_paths(G, graph, root):
    comp = nx.node_connected_component(G, root)
    terms = [n for n in comp if G.degree(n) == 1 and n != root]
    paths = {}
    for t in terms:
        try:
            paths[t] = nx.shortest_path(G, root, t)
        except nx.NetworkXNoPath:
            pass
    print(f"[Info] {len(paths)} paths in main component ({len(comp)} nodes).")
    return paths


def expression_dynamics(graph, adata, paths, genes):
    vn = list(adata.var_names)
    gi = [(g, vn.index(g)) for g in genes if g in vn]
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    dynamics = {}
    for tid, path in paths.items():
        rows = []
        for i, nid in enumerate(path):
            mem = graph['nodes'][nid]
            me = np.mean(X[mem], axis=0)
            row = {g: me[idx] for g, idx in gi}
            row['pos'] = i
            row['node'] = nid
            row['n_cells'] = len(mem)
            rows.append(row)
        dynamics[tid] = pd.DataFrame(rows)
    return dynamics


def plot_dynamics(dynamics, genes, save_path=None):
    n = min(len(genes), 4)
    if n == 0 or not dynamics: return
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    cm = plt.cm.Set2
    for gi, g in enumerate(genes[:n]):
        ax = axes[0][gi]
        for pi, (tid, df) in enumerate(dynamics.items()):
            if g in df.columns:
                ax.plot(df['pos'], df[g], color=cm(pi%8), marker='o',
                        markersize=4, label=f'→{tid[:12]}', linewidth=1.8)
        ax.set_xlabel('Position'); ax.set_ylabel('Expression')
        ax.set_title(g, fontweight='bold')
        if gi == 0: ax.legend(fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================
# 参数稳定性扫描
# ============================================================
def stability_sweep(pca_data, lens_1d, eps_base):
    results = []
    for nc in [6, 8, 10, 15]:
        for ov in [0.4, 0.5, 0.6]:
            for ef in [0.8, 1.0, 1.2]:
                eps = eps_base * ef
                try:
                    kmap = km.KeplerMapper(verbose=0)
                    lens = lens_1d.reshape(-1, 1) if lens_1d.ndim == 1 else lens_1d
                    g = kmap.map(lens, X=pca_data,
                                  cover=km.Cover(n_cubes=nc, perc_overlap=ov),
                                  clusterer=DBSCAN(eps=eps, min_samples=3))
                    G = nx.Graph()
                    for nid, mem in g['nodes'].items(): G.add_node(nid, members=mem)
                    for src, tgts in g['links'].items():
                        for t in tgts: G.add_edge(src, t)
                    comps = list(nx.connected_components(G))
                    n_term = sum(1 for n in G.nodes()
                                 if G.degree(n) <= 1 and len(g['nodes'][n]) < 50)
                    results.append({'n_cubes': nc, 'overlap': ov, 'eps': round(eps, 3),
                                     'n_nodes': G.number_of_nodes(),
                                     'n_comps': len(comps),
                                     'largest': max(len(c) for c in comps) if comps else 0,
                                     'n_rare': n_term})
                except:
                    pass
    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Mapper 拓扑结构与稳定性 (→ 4.2补充, 4.5)")
    print("=" * 60)

    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

    # 1. 节点角色
    print("\n=== Node Roles ===")
    ndf = classify_nodes(graph, G)

    # 2. Mapper 图
    plot_mapper_graph(graph, G, ndf, title="Mapper Topology",
                       save_path=figp("mapper_topology.png"))
    print(f"\n[Info] Saved: {figp('mapper_topology.png')}")

    # 3. 轨迹分析
    print("\n=== Trajectory ===")
    root = identify_root(G)
    paths = find_paths(G, graph, root)

    genes = [g for g in ['Tmem37', 'C1qb', 'Sepp1', 'Blnk', 'Ctss',
                          'Ms4a6c', 'Cd34', 'Kit'] if g in adata.var_names]
    if genes and paths:
        dyn = expression_dynamics(graph, adata, paths, genes)
        plot_dynamics(dyn, genes[:4], save_path=figp("gene_dynamics.png"))
        print(f"[Info] Saved: {figp('gene_dynamics.png')}")

    # 4. 分支点
    bps = [n for n in nx.node_connected_component(G, root) if G.degree(n) >= 3]
    print(f"\n[Info] {len(bps)} branch points in main component.")

    # 5. 稳定性
    print("\n=== Parameter Stability ===")
    eb = auto_eps(adata.obsm['X_pca'])
    stab = stability_sweep(adata.obsm['X_pca'], lens_1d, eb)
    print(stab.describe().to_string())
    ok = (stab['n_rare'] > 0).sum()
    print(f"\n{ok}/{len(stab)} detect rare branches ({ok/len(stab)*100:.0f}%)")

    # 6. HTML
    kmap = km.KeplerMapper(verbose=0)
    kmap.visualize(graph, path_html=figp("topology.html"), title="scRNA Topology")
    print(f"[Info] Saved: {figp('topology.html')}")

    print("\n[Done]")
