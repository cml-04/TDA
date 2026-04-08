"""
exp_ablation.py — 插值消融实验 + Leiden 基准对比
================================================
来源: thesis_analysis.py (模块1 + 模块2)
章节: 4.3 插值消融实验
输出:
  fig/ablation_umap.png           三种插值 UMAP 对比
  fig/ablation_pca_variance.png   PCA 方差解释率
  fig/benchmark_ari_nmi.png       Mapper vs Leiden ARI/NMI
  fig/benchmark_cluster_count.png Mapper vs Leiden 聚类数
"""

import numpy as np
import pandas as pd
import scanpy as sc
import magic
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from gtda.homology import VietorisRipsPersistence
import warnings
warnings.filterwarnings('ignore')

from core import (RANDOM_SEED, DATA_PATH, fig_path, make_fig, ensure_fig_dir,
                  load_and_normalize, full_pipeline)

ensure_fig_dir('ablation')
figp = make_fig('ablation')
sc.settings.verbosity = 1


# ============================================================
# 模块 1: Leiden 基准对比 (Mapper vs Leiden 的 ARI/NMI)
# ============================================================
def benchmark_vs_leiden(adata, mapper_labels):
    print("\n" + "=" * 60)
    print("Leiden 基准对比 (Mapper vs Leiden)")
    print("=" * 60)

    ab = adata.copy()
    if 'neighbors' not in ab.uns:
        sc.pp.neighbors(ab, random_state=RANDOM_SEED)

    resolutions = [0.3, 0.5, 0.8, 1.0, 1.5]
    valid = mapper_labels >= 0
    mapper_str = mapper_labels[valid].astype(str)
    results = []

    for res in resolutions:
        sc.tl.leiden(ab, resolution=res, key_added=f'leiden_{res}',
                     random_state=RANDOM_SEED)
        ll = ab.obs[f'leiden_{res}'].values.astype(str)
        results.append({
            'resolution': res,
            'leiden_k': len(set(ll)),
            'mapper_k': len(set(mapper_str)),
            'ARI': round(adjusted_rand_score(mapper_str, ll[valid]), 4),
            'NMI': round(normalized_mutual_info_score(mapper_str, ll[valid]), 4),
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    # ARI/NMI 折线图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df['resolution'], df['ARI'], 'o-', lw=2, color='#3498db',
                 label='Mapper vs Leiden')
    axes[0].set_xlabel('Resolution'); axes[0].set_ylabel('ARI')
    axes[0].set_title('Adjusted Rand Index'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df['resolution'], df['NMI'], 'o-', lw=2, color='#e67e22',
                 label='Mapper vs Leiden')
    axes[1].set_xlabel('Resolution'); axes[1].set_ylabel('NMI')
    axes[1].set_title('Normalized Mutual Information'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figp('benchmark_ari_nmi.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 聚类数对比柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(resolutions)); w = 0.3
    ax.bar(x - w/2, df['leiden_k'], w, label='Leiden', color='#3498db')
    ax.bar(x + w/2, df['mapper_k'], w, label='Mapper', color='#e74c3c')
    ax.set_xticks(x); ax.set_xticklabels([f'res={r}' for r in resolutions])
    ax.set_ylabel('Clusters'); ax.set_title('Cluster Count Comparison'); ax.legend()
    plt.tight_layout()
    plt.savefig(figp('benchmark_cluster_count.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[Info] Saved: benchmark_ari_nmi.png, benchmark_cluster_count.png")
    return df


# ============================================================
# 模块 2: 插值消融
# ============================================================
def imputation_ablation(path=DATA_PATH, n_hvg=2000):
    print("\n" + "=" * 60)
    print("插值消融实验 (无插值 / KNN / MAGIC)")
    print("=" * 60)

    base = load_and_normalize(path)
    methods = {}

    # A: 无插值
    print("\n[A] No imputation")
    a = base.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor='seurat', subset=True)
    sc.tl.pca(a, n_comps=30, random_state=RANDOM_SEED)
    sc.pp.neighbors(a, random_state=RANDOM_SEED)
    sc.tl.umap(a, random_state=RANDOM_SEED)
    sc.tl.leiden(a, resolution=0.5, random_state=RANDOM_SEED)
    methods['No Imputation'] = a

    # B: KNN
    print("[B] KNN imputation")
    b = base.copy()
    Xb = b.X.toarray() if issparse(b.X) else b.X.copy()
    Xb[Xb == 0] = np.nan
    Xb = np.maximum(KNNImputer(n_neighbors=10).fit_transform(Xb), 0)
    b.X = Xb
    sc.pp.highly_variable_genes(b, n_top_genes=n_hvg, flavor='seurat', subset=True)
    sc.tl.pca(b, n_comps=30, random_state=RANDOM_SEED)
    sc.pp.neighbors(b, random_state=RANDOM_SEED)
    sc.tl.umap(b, random_state=RANDOM_SEED)
    sc.tl.leiden(b, resolution=0.5, random_state=RANDOM_SEED)
    methods['KNN'] = b

    # C: MAGIC
    print("[C] MAGIC imputation")
    c = base.copy()
    if issparse(c.X): c.X = np.expm1(c.X.toarray())
    else: c.X = np.expm1(c.X)
    c.X = np.sqrt(c.X)
    c.X = magic.MAGIC(t=3, n_pca=20, n_jobs=-1, random_state=RANDOM_SEED
                       ).fit_transform(c.to_df()).values
    sc.pp.highly_variable_genes(c, n_top_genes=n_hvg, flavor='seurat', subset=True)
    sc.tl.pca(c, n_comps=30, random_state=RANDOM_SEED)
    sc.pp.neighbors(c, random_state=RANDOM_SEED)
    sc.tl.umap(c, random_state=RANDOM_SEED)
    sc.tl.leiden(c, resolution=0.5, random_state=RANDOM_SEED)
    methods['MAGIC'] = c

    # === 对比统计 ===
    pca_var = {}
    print("\nPCA variance explained (top 5):")
    for nm, ad in methods.items():
        v = ad.uns['pca']['variance_ratio'][:10]
        pca_var[nm] = v
        print(f"  {nm}: {v[:5].sum()*100:.1f}%")

    print("\nZero fraction:")
    for nm, ad in methods.items():
        X = ad.X.toarray() if issparse(ad.X) else ad.X
        print(f"  {nm}: {(X==0).sum()/X.size*100:.1f}%")

    print("\nLeiden clusters:")
    for nm, ad in methods.items():
        print(f"  {nm}: {len(ad.obs['leiden'].unique())}")

    print("\nPersistent homology (H0):")
    vr = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
    for nm, ad in methods.items():
        diag = vr.fit_transform(ad.obsm['X_pca'][np.newaxis, :, :])[0]
        h0l = diag[diag[:, 2] == 0][:, 1] - diag[diag[:, 2] == 0][:, 0]
        n_sig = int((h0l > np.median(h0l) * 3).sum())
        print(f"  {nm}: {n_sig} significant H0 bars, max lifetime={h0l.max():.1f}")

    print("\nPairwise Leiden ARI:")
    nms = list(methods.keys())
    for i in range(len(nms)):
        for j in range(i+1, len(nms)):
            ari = adjusted_rand_score(
                methods[nms[i]].obs['leiden'].values.astype(str),
                methods[nms[j]].obs['leiden'].values.astype(str))
            print(f"  {nms[i]} vs {nms[j]}: {ari:.4f}")

    # === 绘图 ===
    # UMAP 三合一
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nm, ad) in zip(axes, methods.items()):
        sc.pl.umap(ad, color='leiden', ax=ax, show=False, title=nm,
                    legend_loc='on data', legend_fontsize=8)
    plt.tight_layout()
    plt.savefig(figp('ablation_umap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # PCA 方差
    fig, ax = plt.subplots(figsize=(8, 5))
    for nm, v in pca_var.items():
        ax.plot(range(1, 11), v[:10]*100, 'o-', lw=2, label=nm)
    ax.set_xlabel('PC'); ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Variance by Imputation Method'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figp('ablation_pca_variance.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Info] Saved: ablation_umap.png, ablation_pca_variance.png")
    return methods


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("插值消融实验 + Leiden 基准 (→ 4.3)")
    print("=" * 60)

    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

    bench_df = benchmark_vs_leiden(adata, mapper_labels)
    ablation_methods = imputation_ablation()

    print("\n[Done]")
