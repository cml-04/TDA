"""
exp_figures.py — 论文统一出图 + Excel 导出
==========================================
来源: thesis_analysis.py (模块5 + Excel)
章节: 全章通用
输出:
  fig/fig_umap_groups.png     UMAP 按 Mapper 群体着色
  fig/fig_umap_markers.png    UMAP marker 基因表达
  fig/fig_mapper_graph.png    Mapper 拓扑图 (按连通分量着色)
  fig/fig_overview_panel.png  综合 2×3 面板
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from matplotlib.lines import Line2D

from core import (RANDOM_SEED, fig_path, make_fig, full_pipeline, ensure_fig_dir)

ensure_fig_dir('figures')
figp = make_fig('figures')
sc.settings.verbosity = 1


# ============================================================
# UMAP 群体图
# ============================================================
def plot_umap_groups(adata, mapper_labels, save_path=None):
    adata.obs['mapper_group'] = pd.Categorical(
        [f'Group_{l}' if l >= 0 else 'Unassigned' for l in mapper_labels])
    fig, ax = plt.subplots(figsize=(8, 7))
    sc.pl.umap(adata, color='mapper_group', ax=ax, show=False,
                title='Cell Groups (Mapper)', frameon=True, legend_fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# UMAP marker 基因
# ============================================================
def plot_umap_markers(adata, save_path=None):
    if 'mapper_group' not in adata.obs:
        return
    sc.tl.rank_genes_groups(adata, 'mapper_group', method='wilcoxon', use_raw=True)
    groups = [g for g in adata.obs['mapper_group'].cat.categories if g != 'Unassigned']
    key_genes = []
    for g in groups:
        try:
            df = sc.get.rank_genes_groups_df(adata, group=g)
            sig = df[df['pvals_adj'] < 0.05]
            if len(sig) > 0:
                gene = sig.iloc[0]['names']
                if adata.raw is not None and gene in adata.raw.var_names:
                    key_genes.append(gene)
        except:
            pass
    key_genes = list(dict.fromkeys(key_genes))[:6]
    if not key_genes:
        print("[Warning] No significant marker genes found.")
        return key_genes

    ng = len(key_genes); nc = min(3, ng); nr = (ng + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if ng == 1: axes = np.array([axes])
    af = axes.flatten()
    for i, gene in enumerate(key_genes):
        sc.pl.umap(adata, color=gene, ax=af[i], show=False,
                    use_raw=True, cmap='viridis', frameon=True)
    for j in range(i+1, len(af)):
        af[j].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return key_genes


# ============================================================
# Mapper 拓扑图 (按连通分量着色)
# ============================================================
def plot_mapper_components(graph, G, comps, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=1.5, iterations=100)
    cv = plt.cm.Set2(np.linspace(0, 1, max(len(comps), 1)))
    cc = {}
    for ci, comp in enumerate(comps):
        for nid in comp:
            cc[nid] = cv[ci]
    colors = [cc.get(n, [.7,.7,.7,1]) for n in G.nodes()]
    sizes = [len(graph['nodes'][n]) * 4 + 40 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                            alpha=0.9, edgecolors='white', linewidths=0.5)
    for nid in G.nodes():
        x, y = pos[nid]; s = len(graph['nodes'][nid])
        if s >= 20 or G.degree(nid) >= 3:
            ax.annotate(f'{s}', (x, y), fontsize=7, ha='center', va='center',
                        fontweight='bold')
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor=cv[i],
               markersize=12, label=f'Comp {i} ({len(comps[i])} nodes)')
        for i in range(min(len(comps), 8))
    ], loc='upper left', fontsize=9)
    ax.set_title('Mapper Graph (size ∝ cells)', fontsize=14)
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return pos, colors, sizes


# ============================================================
# 综合面板 (2×3)
# ============================================================
def plot_overview_panel(adata, graph, G, comps, pos, colors, sizes,
                         key_genes=None, save_path=None):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

    # (A) UMAP groups
    ax_a = fig.add_subplot(gs[0, 0])
    sc.pl.umap(adata, color='mapper_group', ax=ax_a, show=False,
                title='(A) Mapper Groups', frameon=True, legend_fontsize=7)

    # (B) Leiden
    if 'leiden' not in adata.obs:
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, random_state=RANDOM_SEED)
        sc.tl.leiden(adata, resolution=0.5, random_state=RANDOM_SEED)
    ax_b = fig.add_subplot(gs[0, 1])
    sc.pl.umap(adata, color='leiden', ax=ax_b, show=False,
                title='(B) Leiden', frameon=True, legend_fontsize=7, legend_loc='on data')

    # (C) Mapper graph
    ax_c = fig.add_subplot(gs[0, 2])
    nx.draw_networkx_edges(G, pos, ax=ax_c, alpha=0.4, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax_c, node_color=colors,
                            node_size=[s*0.7 for s in sizes], alpha=0.9,
                            edgecolors='white', linewidths=0.3)
    ax_c.set_title('(C) Mapper Graph'); ax_c.axis('off')

    # (D-F) marker 基因
    if key_genes:
        for gi, gene in enumerate(key_genes[:3]):
            ax = fig.add_subplot(gs[1, gi])
            sc.pl.umap(adata, color=gene, ax=ax, show=False, use_raw=True,
                        cmap='viridis', frameon=True, title=f'({chr(68+gi)}) {gene}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("论文统一出图 (→ 全章)")
    print("=" * 60)

    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata, random_state=RANDOM_SEED)
        sc.tl.umap(adata, random_state=RANDOM_SEED)

    # 1. UMAP groups
    plot_umap_groups(adata, mapper_labels, save_path=figp('fig_umap_groups.png'))
    print(f"[Info] Saved: {figp('fig_umap_groups.png')}")

    # 2. UMAP markers
    key_genes = plot_umap_markers(adata, save_path=figp('fig_umap_markers.png'))
    print(f"[Info] Saved: {figp('fig_umap_markers.png')}")

    # 3. Mapper graph
    pos, colors, sizes = plot_mapper_components(
        graph, G, comps, save_path=figp('fig_mapper_graph.png'))
    print(f"[Info] Saved: {figp('fig_mapper_graph.png')}")

    # 4. Overview panel
    plot_overview_panel(adata, graph, G, comps, pos, colors, sizes,
                         key_genes=key_genes,
                         save_path=figp('fig_overview_panel.png'))
    print(f"[Info] Saved: {figp('fig_overview_panel.png')}")

    print("\n[Done]")
