"""
exp_edge_expression.py — 边缘节点表达谱比较
=============================================
对 Mapper rarity score 最高的边缘节点, 比较其基因表达与:
  (1) 所在连通分量的均值
  (2) 全局均值
验证拓扑边缘性是否具有生物学意义

章节: 4.5 拓扑边缘细胞分析 (核心支撑)
输出:
  fig/edge_expression/edge_heatmap.png        多层级 Z-score 热力图
  fig/edge_expression/edge_profile.png        表达谱折线对比 (节点 vs 分量 vs 全局)
  fig/edge_expression/edge_boxplot.png        关键基因箱线图 (三组对比)
  fig/edge_expression/edge_volcano.png        边缘 vs 分量的 DE 火山图
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

from core import (RANDOM_SEED, PROJECT_ROOT, make_fig, ensure_fig_dir,
                  full_pipeline, assign_mapper_labels)

figp = make_fig('edge_expression')
ensure_fig_dir('edge_expression')
sc.settings.verbosity = 0
np.random.seed(RANDOM_SEED)

TOP_N_NODES = 5       # 分析前 N 个最稀有的边缘节点
TOP_N_GENES = 20      # 每个节点的 top DE 基因数
MIN_NODE_CELLS = 3    # 节点最少细胞数


# ============================================================
# 1. 计算 Rarity Score (与 exp_rarity.py 一致)
# ============================================================
def compute_rarity(graph, G, adata):
    artic = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if len(sub) > 2:
            artic.update(nx.articulation_points(sub))

    hubs = [n for n in G.nodes() if G.degree(n) >= 3]
    if not hubs:
        hubs = [max(G.nodes(), key=lambda n: G.degree(n))]

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    gm = np.mean(X, axis=0)

    records = []
    for n in G.nodes():
        mem = graph['nodes'][n]
        d = G.degree(n)
        if d == 0: role = 'Isolated'
        elif d == 1: role = 'Terminal'
        elif n in artic: role = 'Bridge'
        elif d >= 3: role = 'Hub'
        else: role = 'Chain'

        md = min((nx.shortest_path_length(G, n, h)
                  for h in hubs if nx.has_path(G, n, h)), default=999)
        nbs = [len(graph['nodes'][nb]) for nb in G.neighbors(n)]
        dr = len(mem) / (np.mean(nbs) + 1e-6) if nbs else 0.5
        td = np.linalg.norm(np.mean(X[mem], axis=0) - gm)

        # 所在连通分量
        comp_id = -1
        for ci, comp in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
            if n in comp:
                comp_id = ci
                break

        records.append({
            'node_id': n, 'n_cells': len(mem), 'role': role,
            'degree': d, 'topo_iso': md, 'density_ratio': dr,
            'trans_dist': td, 'comp_id': comp_id
        })

    df = pd.DataFrame(records)
    for c in ['topo_iso', 'trans_dist']:
        mn, mx = df[c].min(), df[c].max()
        df[f'{c}_n'] = (df[c] - mn) / (mx - mn + 1e-8)
    mn, mx = df['density_ratio'].min(), df['density_ratio'].max()
    df['density_n'] = 1.0 - (df['density_ratio'] - mn) / (mx - mn + 1e-8)
    median_size = df['n_cells'].median()
    df['size_penalty'] = np.clip(df['n_cells'] / (median_size + 1e-12), 0, 1)
    df['rarity'] = (0.3 * df['topo_iso_n'] + 0.2 * df['density_n'] +
                     0.2 * df['trans_dist_n'] + 0.3 * (1.0 - df['size_penalty']))
    return df.sort_values('rarity', ascending=False)


# ============================================================
# 2. 获取表达矩阵 (用 raw log-normalized 数据)
# ============================================================
def get_expression(adata, indices, gene_names=None):
    """提取指定细胞的表达矩阵"""
    if adata.raw is not None:
        X = adata.raw.X[indices]
        vn = list(adata.raw.var_names)
    else:
        X = adata.X[indices]
        vn = list(adata.var_names)
    if issparse(X): X = X.toarray()
    if gene_names:
        gi = [vn.index(g) for g in gene_names if g in vn]
        gn = [g for g in gene_names if g in vn]
        return X[:, gi], gn
    return X, vn


# ============================================================
# 3. 找边缘节点的 DE 基因
# ============================================================
def find_edge_de_genes(adata, graph, comps_sorted, rarity_df, top_n_nodes=TOP_N_NODES):
    """对每个边缘节点做 Wilcoxon DE vs 所在分量其余细胞"""
    # 筛选: 小节点 + 高 rarity
    candidates = rarity_df[rarity_df['n_cells'] >= MIN_NODE_CELLS].head(top_n_nodes * 2)
    selected = candidates.head(top_n_nodes)

    results = []
    for _, row in selected.iterrows():
        nid = row['node_id']
        comp_id = row['comp_id']
        node_cells = graph['nodes'][nid]

        # 分量中除该节点外的所有细胞
        comp_nodes = comps_sorted[comp_id]
        comp_cells = []
        for cn in comp_nodes:
            comp_cells.extend(graph['nodes'][cn])
        comp_other = [c for c in comp_cells if c not in node_cells]

        if len(comp_other) < 10:
            continue

        # Wilcoxon DE
        labels = np.array(['Other'] * adata.n_obs, dtype=object)
        for c in node_cells:
            if c < adata.n_obs:
                labels[c] = 'Edge'
        adata.obs['_edge_group'] = pd.Categorical(labels)

        try:
            sc.tl.rank_genes_groups(adata, '_edge_group', groups=['Edge'],
                                     reference='Other', method='wilcoxon',
                                     use_raw=True)
            de = sc.get.rank_genes_groups_df(adata, group='Edge')
            de_sig = de[de['pvals_adj'] < 0.05].head(TOP_N_GENES)
        except:
            de_sig = pd.DataFrame()

        results.append({
            'node_id': nid, 'n_cells': len(node_cells),
            'role': row['role'], 'rarity': row['rarity'],
            'comp_id': comp_id, 'n_comp_cells': len(comp_cells),
            'de_genes': de_sig,
            'node_cells': node_cells,
            'comp_cells': comp_cells,
            'comp_other': comp_other,
        })
        print(f"  Node {nid}: {len(node_cells)} cells, rarity={row['rarity']:.3f}, "
              f"role={row['role']}, comp={comp_id} ({len(comp_cells)} cells), "
              f"DE genes={len(de_sig)}")

    return results


# ============================================================
# 图1: 多层级热力图
# ============================================================
def plot_edge_heatmap(adata, edge_results, graph, comps_sorted):
    """行 = 各边缘节点 + 各分量均值 + 全局均值, 列 = top DE 基因"""
    print("\n[图1] Edge Heatmap")

    # 收集所有 DE 基因 (去重)
    all_genes = []
    gene_source = {}
    for er in edge_results:
        for _, g in er['de_genes'].iterrows():
            gene = g['names']
            if gene not in gene_source:
                all_genes.append(gene)
                gene_source[gene] = er['node_id']
    all_genes = all_genes[:40]  # 限制

    if not all_genes:
        print("  ⚠ 无显著DE基因，跳过")
        return

    # 构建表达矩阵: 每行一个"组"
    row_labels = []
    row_means = []

    # 全局均值
    X_global, gn = get_expression(adata, list(range(adata.n_obs)), all_genes)
    row_labels.append('Global Mean')
    row_means.append(np.mean(X_global, axis=0))

    # 各连通分量均值
    seen_comps = set()
    for er in edge_results:
        cid = er['comp_id']
        if cid not in seen_comps:
            seen_comps.add(cid)
            X_comp, _ = get_expression(adata, er['comp_cells'], all_genes)
            row_labels.append(f'Comp_{cid} Mean (n={len(er["comp_cells"])})')
            row_means.append(np.mean(X_comp, axis=0))

    # 各边缘节点均值
    for er in edge_results:
        X_node, _ = get_expression(adata, er['node_cells'], all_genes)
        row_labels.append(f'Edge {er["node_id"][:15]} '
                          f'(n={er["n_cells"]}, r={er["rarity"]:.2f})')
        row_means.append(np.mean(X_node, axis=0))

    mat = np.array(row_means)
    # Z-score 按列
    col_mean = mat.mean(axis=0)
    col_std = mat.std(axis=0) + 1e-8
    zmat = (mat - col_mean) / col_std

    # 绘图
    n_rows = len(row_labels)
    n_cols = len(all_genes)
    fig_h = max(4, n_rows * 0.5 + 2)
    fig_w = max(12, n_cols * 0.4 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = min(np.abs(zmat).max(), 3)
    im = ax.imshow(zmat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_cols))
    display_genes = all_genes[:n_cols]
    ax.set_xticklabels(display_genes, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # 分隔线: 全局 | 分量们 | 边缘们
    n_global = 1
    n_comp = len(seen_comps)
    ax.axhline(n_global - 0.5, color='black', lw=2)
    ax.axhline(n_global + n_comp - 0.5, color='black', lw=2)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Z-score')
    ax.set_title('Edge Node Expression vs Component vs Global\n(Z-scored per gene)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('edge_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('edge_heatmap.png')}")


# ============================================================
# 图2: 表达谱折线对比
# ============================================================
def plot_edge_profiles(adata, edge_results):
    """每个边缘节点: 折线图 node vs comp vs global"""
    print("\n[图2] Edge Profiles")

    n_panels = min(len(edge_results), 3)
    if n_panels == 0: return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1: axes = [axes]

    for ax, er in zip(axes, edge_results[:n_panels]):
        genes = er['de_genes']['names'].head(12).tolist()
        if not genes: continue

        # 三层均值
        X_node, gn = get_expression(adata, er['node_cells'], genes)
        X_comp, _ = get_expression(adata, er['comp_other'], genes)
        X_global, _ = get_expression(adata, list(range(adata.n_obs)), genes)

        x = np.arange(len(gn))
        ax.plot(x, np.mean(X_node, axis=0), 'o-', color='#e74c3c', lw=2.5,
                ms=7, label=f'Edge node (n={er["n_cells"]})', zorder=3)
        ax.plot(x, np.mean(X_comp, axis=0), 's--', color='#3498db', lw=1.5,
                ms=5, label=f'Comp_{er["comp_id"]} others (n={len(er["comp_other"])})')
        ax.plot(x, np.mean(X_global, axis=0), '^:', color='#95a5a6', lw=1,
                ms=4, label=f'Global (n={adata.n_obs})')

        ax.set_xticks(x)
        ax.set_xticklabels(gn, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Expression (log-normalized)')
        ax.set_title(f'Edge {er["node_id"][:15]}\n'
                     f'rarity={er["rarity"]:.2f}, {er["role"]}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Edge Node Expression Profile vs Component vs Global',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('edge_profile.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('edge_profile.png')}")


# ============================================================
# 图3: 关键基因箱线图
# ============================================================
def plot_edge_boxplots(adata, edge_results):
    """选最稀有的1个节点, 画其 top 6 DE 基因的三组箱线图"""
    print("\n[图3] Edge Boxplots")

    if not edge_results: return
    er = edge_results[0]  # 最稀有的节点
    genes = er['de_genes']['names'].head(6).tolist()
    if not genes: return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for gi, gene in enumerate(genes):
        if gi >= 6: break
        ax = axes[gi]

        X_node, gn = get_expression(adata, er['node_cells'], [gene])
        X_comp, _ = get_expression(adata, er['comp_other'], [gene])
        X_global, _ = get_expression(adata, list(range(adata.n_obs)), [gene])

        data = [X_node[:, 0], X_comp[:, 0], X_global[:, 0]]
        labels = [f'Edge\n(n={len(er["node_cells"])})',
                  f'Comp_{er["comp_id"]}\nothers\n(n={len(er["comp_other"])})',
                  f'Global\n(n={adata.n_obs})']
        colors = ['#e74c3c', '#3498db', '#95a5a6']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        showfliers=True, flierprops=dict(markersize=2, alpha=0.3))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Mann-Whitney p-value (edge vs comp)
        if len(X_node[:, 0]) >= 3 and len(X_comp[:, 0]) >= 3:
            try:
                _, pval = mannwhitneyu(X_node[:, 0], X_comp[:, 0], alternative='two-sided')
                sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else
                       ('*' if pval < 0.05 else 'ns'))
                ax.text(0.95, 0.95, f'p={pval:.2e} {sig}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except:
                pass

        ax.set_title(gene, fontsize=11, fontweight='bold')
        ax.set_ylabel('Expression')
        ax.grid(axis='y', alpha=0.3)

    for j in range(len(genes), 6):
        axes[j].axis('off')

    plt.suptitle(f'Edge Node {er["node_id"][:15]} (rarity={er["rarity"]:.2f}, '
                 f'{er["role"]}, n={er["n_cells"]})\n'
                 f'Top DE Genes: Edge vs Component vs Global',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('edge_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('edge_boxplot.png')}")


# ============================================================
# 图4: Volcano Plot (edge vs component)
# ============================================================
def plot_edge_volcano(adata, edge_results):
    """最稀有节点 vs 所在分量的 DE 火山图"""
    print("\n[图4] Edge Volcano Plot")

    if not edge_results: return
    er = edge_results[0]

    de = er['de_genes']
    if len(de) == 0:
        # 重新做全基因组 DE
        labels = np.array(['Other'] * adata.n_obs, dtype=object)
        for c in er['node_cells']:
            if c < adata.n_obs:
                labels[c] = 'Edge'
        adata.obs['_volcano'] = pd.Categorical(labels)
        sc.tl.rank_genes_groups(adata, '_volcano', groups=['Edge'],
                                 reference='Other', method='wilcoxon', use_raw=True)
        de = sc.get.rank_genes_groups_df(adata, group='Edge')

    if len(de) == 0:
        print("  ⚠ 无DE结果，跳过")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    logfc = de['logfoldchanges'].values
    pval = de['pvals_adj'].values
    neg_log_p = -np.log10(pval + 1e-300)
    genes = de['names'].values

    # 分类: 显著上调 / 显著下调 / 不显著
    sig_up = (pval < 0.05) & (logfc > 0.5)
    sig_down = (pval < 0.05) & (logfc < -0.5)
    ns = ~sig_up & ~sig_down

    ax.scatter(logfc[ns], neg_log_p[ns], c='#cccccc', s=8, alpha=0.5, label='Not sig.')
    ax.scatter(logfc[sig_up], neg_log_p[sig_up], c='#e74c3c', s=20, alpha=0.7,
               label=f'Up in edge ({sig_up.sum()})')
    ax.scatter(logfc[sig_down], neg_log_p[sig_down], c='#3498db', s=20, alpha=0.7,
               label=f'Down in edge ({sig_down.sum()})')

    # 标注 top 基因
    top_mask = (sig_up | sig_down) & (neg_log_p > np.percentile(neg_log_p[pval < 0.05], 80)
                                       if (pval < 0.05).sum() > 5 else np.ones(len(pval), bool))
    for i in np.where(top_mask)[0][:15]:
        ax.annotate(genes[i], (logfc[i], neg_log_p[i]),
                    fontsize=7, alpha=0.8, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5))

    ax.axhline(-np.log10(0.05), color='gray', ls='--', alpha=0.5, label='p=0.05')
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.axvline(-0.5, color='gray', ls=':', alpha=0.3)

    ax.set_xlabel('log₂ Fold Change (Edge vs Component)', fontsize=12)
    ax.set_ylabel('-log₁₀ adjusted p-value', fontsize=12)
    ax.set_title(f'Volcano Plot: Edge Node {er["node_id"][:15]}\n'
                 f'({er["n_cells"]} cells, rarity={er["rarity"]:.2f}) '
                 f'vs Comp_{er["comp_id"]} ({len(er["comp_other"])} cells)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(figp('edge_volcano.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('edge_volcano.png')}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("边缘节点表达谱比较 (→ 4.5)")
    print("=" * 65)

    # 加载
    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()
    comps_sorted = sorted(nx.connected_components(G), key=len, reverse=True)

    # Ground Truth (如果有)
    gt_path = os.path.join(PROJECT_ROOT, 'cell_type_labels.csv')
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path).set_index('cell_id')
        common = adata.obs_names.intersection(gt.index)
        gt_map = gt.loc[common, 'cell_type'].to_dict()
    else:
        gt_map = {}

    # 计算 rarity
    print("\n=== Rarity Scores ===")
    rdf = compute_rarity(graph, G, adata)
    print(rdf[['node_id', 'n_cells', 'role', 'rarity', 'comp_id']].head(10).to_string())

    # 找 DE 基因
    print("\n=== Edge Node DE Analysis ===")
    edge_results = find_edge_de_genes(adata, graph, comps_sorted, rdf)

    if not edge_results:
        print("\n⚠ 未找到合适的边缘节点，退出")
        exit()

    # 打印边缘节点的 ground truth 组成
    if gt_map:
        print("\n=== Edge Node Cell Type Composition ===")
        for er in edge_results:
            cells = [adata.obs_names[c] for c in er['node_cells'] if c < adata.n_obs]
            types = [gt_map.get(c, 'Unknown') for c in cells]
            tc = pd.Series(types).value_counts()
            print(f"\n  Node {er['node_id'][:15]} (n={er['n_cells']}, "
                  f"rarity={er['rarity']:.2f}, Comp_{er['comp_id']}):")
            for t, n in tc.items():
                print(f"    {t}: {n} ({n/len(types)*100:.0f}%)")

    # 生成 4 张图
    plot_edge_heatmap(adata, edge_results, graph, comps_sorted)
    plot_edge_profiles(adata, edge_results)
    plot_edge_boxplots(adata, edge_results)
    plot_edge_volcano(adata, edge_results)

    print(f"\n{'=' * 65}")
    print("输出:")
    print(f"  fig/edge_expression/edge_heatmap.png   多层级热力图")
    print(f"  fig/edge_expression/edge_profile.png   表达谱折线对比")
    print(f"  fig/edge_expression/edge_boxplot.png   关键基因箱线图")
    print(f"  fig/edge_expression/edge_volcano.png   DE 火山图")
    print("=" * 65)
    print("[Done]")