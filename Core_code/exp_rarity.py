"""
exp_rarity.py — 稀有/边缘细胞分析
===================================
来源: main.py (RareCellIdentifier) + thesis_analysis.py (模块4)
章节: 4.5 拓扑边缘细胞分析
输出:
  fig/rare_cells.png       Mapper rarity score 可视化
  fig/rare_comparison.png  三种方法 (Mapper/Gini/Density) 对比
  控制台输出 rarity ranking, DE, Jaccard overlap
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from core import (RANDOM_SEED, fig_path, make_fig, full_pipeline, ensure_fig_dir)

ensure_fig_dir('rarity')
figp = make_fig('rarity')


# ============================================================
# Mapper Rarity Score
# ============================================================
def score_rarity(graph, G, adata):
    """4维度复合 rarity score: topo_iso + density + trans_dist + size"""
    # 节点角色
    artic = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if len(sub) > 2:
            artic.update(nx.articulation_points(sub))
    ndf_records = []
    for n in G.nodes():
        d = G.degree(n)
        s = len(graph['nodes'][n])
        if d == 0: r = 'Isolated'
        elif d == 1: r = 'Terminal'
        elif n in artic: r = 'Bridge'
        elif d >= 3: r = 'Hub'
        else: r = 'Chain'
        ndf_records.append({'node_id': n, 'degree': d, 'n_cells': s, 'role': r})
    ndf = pd.DataFrame(ndf_records)

    # Hub 节点
    hubs = ndf[ndf['role'] == 'Hub']['node_id'].tolist()
    if not hubs:
        hubs = [max(G.nodes(), key=lambda n: G.degree(n))]

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    gm = np.mean(X, axis=0)

    recs = []
    for _, row in ndf.iterrows():
        nid = row['node_id']
        mem = graph['nodes'][nid]
        md = min((nx.shortest_path_length(G, nid, h)
                  for h in hubs if nx.has_path(G, nid, h)), default=999)
        nbs = [len(graph['nodes'][nb]) for nb in G.neighbors(nid)]
        dr = len(mem) / (np.mean(nbs) + 1e-6) if nbs else 0.5
        td = np.linalg.norm(np.mean(X[mem], axis=0) - gm)
        recs.append({'node_id': nid, 'n_cells': len(mem), 'role': row['role'],
                      'degree': row['degree'], 'topo_iso': md,
                      'density_ratio': dr, 'trans_dist': td})

    df = pd.DataFrame(recs)
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


def get_rare_members(graph, node_ids):
    """收集多个节点中的所有细胞索引"""
    idx = []
    for n in node_ids:
        if n in graph['nodes']:
            idx.extend(graph['nodes'][n])
    return np.unique(idx)


# ============================================================
# 三种方法对比 (Mapper / Gini / Density)
# ============================================================
def rare_cell_comparison(adata, mapper_labels, save_path=None):
    """均衡数据上 3 种稀有细胞检测方法的 Jaccard 对比"""
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    nc = adata.n_obs

    # Mapper: 小分量 (< 5%)
    mapper_mask = np.zeros(nc, dtype=bool)
    for lb in np.unique(mapper_labels):
        if lb < 0: continue
        m = mapper_labels == lb
        if m.sum() / nc < 0.05:
            mapper_mask[m] = True
    print(f"\n[Mapper]  {mapper_mask.sum()} cells ({mapper_mask.sum()/nc*100:.1f}%)")

    # Gini: 高不均匀基因空间中的小簇
    def gini(x):
        x = np.sort(x); n = len(x)
        if n == 0 or x.sum() == 0: return 0
        return (2 * np.sum(np.arange(1, n+1) * x) - (n+1) * x.sum()) / (n * x.sum())

    gi = np.array([gini(X[:, j]) for j in range(X.shape[1])])
    hg = X[:, gi > np.percentile(gi, 95)]
    k = max(10, len(set(mapper_labels[mapper_labels >= 0])) * 2)
    gl = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(hg)
    gini_mask = np.zeros(nc, dtype=bool)
    for c in range(k):
        m = gl == c
        if m.sum() / nc < 0.05:
            gini_mask[m] = True
    print(f"[Gini]    {gini_mask.sum()} cells ({gini_mask.sum()/nc*100:.1f}%)")

    # Density: PCA 空间低密度 top 5%
    pca = adata.obsm.get('X_pca', X)
    nn = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(pca)
    d, _ = nn.kneighbors(pca)
    md = d[:, 1:].mean(axis=1)
    dens_mask = md > np.percentile(md, 95)
    print(f"[Density] {dens_mask.sum()} cells ({dens_mask.sum()/nc*100:.1f}%)")

    # Jaccard overlap
    masks = {'Mapper': mapper_mask, 'Gini': gini_mask, 'Density': dens_mask}
    overlap = pd.DataFrame(index=masks, columns=masks, dtype=float)
    for m1, v1 in masks.items():
        for m2, v2 in masks.items():
            u = (v1 | v2).sum()
            overlap.loc[m1, m2] = round((v1 & v2).sum() / u, 3) if u else 0
    print(f"\nJaccard overlap:\n{overlap}")
    print(f"\n  Mapper ∩ Gini:    {(mapper_mask & gini_mask).sum()}")
    print(f"  Mapper ∩ Density: {(mapper_mask & dens_mask).sum()}")
    print(f"  Gini ∩ Density:   {(gini_mask & dens_mask).sum()}")
    print(f"  All three:        {(mapper_mask & gini_mask & dens_mask).sum()}")

    # UMAP 可视化
    umap = adata.obsm.get('X_umap')
    if umap is not None and save_path:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (nm, msk) in zip(axes, masks.items()):
            ax.scatter(umap[~msk, 0], umap[~msk, 1], c='lightgray', s=5, alpha=0.5)
            ax.scatter(umap[msk, 0], umap[msk, 1], c='red', s=15, alpha=0.8,
                       label=f'Rare (n={msk.sum()})')
            ax.set_title(f'{nm}: {msk.sum()} rare cells')
            ax.legend(markerscale=2)
            ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    return overlap


# ============================================================
# DE 分析 (稀有 vs 其他)
# ============================================================
def rare_de(adata, rare_idx, use_raw=True):
    """Wilcoxon DE: rare vs other"""
    labels = np.array(['Other'] * adata.n_obs, dtype=object)
    labels[rare_idx] = 'Rare'
    adata.obs['rare_group'] = pd.Categorical(labels)
    sc.tl.rank_genes_groups(adata, 'rare_group', groups=['Rare'],
                             reference='Other', method='wilcoxon', use_raw=use_raw)
    return sc.get.rank_genes_groups_df(adata, group='Rare')


def check_artifacts(adata, rare_idx):
    """检查稀有细胞是否为 QC 伪影"""
    obs = adata.obs
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[rare_idx] = True
    rpt = {}
    for m in ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']:
        if m in obs:
            rv, ov = obs[m].values[mask], obs[m].values[~mask]
            if np.median(rv) == 0 and np.median(ov) == 0:
                rpt[m] = {'rare': 0, 'other': 0, 'ratio': 1.0, 'suspicious': False}
                continue
            r = np.median(rv) / (np.median(ov) + 1e-8)
            rpt[m] = {'rare': np.median(rv), 'other': np.median(ov),
                       'ratio': r, 'suspicious': abs(r - 1) > 0.3}
    return rpt


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("稀有/边缘细胞分析 (→ 4.5)")
    print("=" * 60)

    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

    # 1. Rarity scoring
    print("\n=== Mapper Rarity Scores ===")
    rdf = score_rarity(graph, G, adata)
    print(rdf[['node_id', 'n_cells', 'role', 'degree', 'rarity']].head(10).to_string())

    # 2. DE for top rare nodes
    top5 = rdf[rdf['n_cells'] <= 50].head(5)['node_id'].tolist()
    ridx = get_rare_members(graph, top5)
    print(f"\n[Info] {len(ridx)} rare cells from top 5 nodes.")

    if len(ridx) >= 5:
        art = check_artifacts(adata, ridx)
        print("\n=== Artifact Check ===")
        for m, v in art.items():
            st = "⚠" if v['suspicious'] else "✓"
            print(f"  {st} {m}: rare={v['rare']:.0f} other={v['other']:.0f} "
                  f"ratio={v['ratio']:.2f}")

        de = rare_de(adata, ridx, use_raw=True)
        sig = de[de['pvals_adj'] < 0.05]
        print(f"\n=== DE: {len(sig)} significant (padj<0.05) ===")
        print(de[['names', 'logfoldchanges', 'pvals_adj']].head(15).to_string())

    # 3. Three-method comparison
    print("\n=== Three-Method Comparison ===")
    overlap = rare_cell_comparison(adata, mapper_labels,
                                    save_path=figp('rare_comparison.png'))
    print(f"\n[Info] Saved: {figp('rare_comparison.png')}")

    print("\n[Done]")
