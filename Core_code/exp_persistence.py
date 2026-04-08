"""
exp_persistence.py — 持久同调分析 + 聚类数估计
===============================================
来源: main.py (TDAAnalyzer) + thesis_analysis.py (模块3)
章节: 4.1 持久同调分析
输出:
  fig/barcode.png          H0/H1 persistence barcode
  fig/cluster_number.png   聚类数定量确定 (3种方法+共识)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence

from core import RANDOM_SEED, fig_path, make_fig, full_pipeline, ensure_fig_dir

ensure_fig_dir('persistence')
figp = make_fig('persistence')


# ============================================================
# 持久同调计算
# ============================================================
def compute_persistence(adata, embedding_key='X_pca', max_dim=2):
    """VR 持久同调"""
    cloud = adata.obsm[embedding_key]
    vr = VietorisRipsPersistence(homology_dimensions=list(range(max_dim)), n_jobs=-1)
    diagrams = vr.fit_transform(cloud[np.newaxis, :, :])
    return diagrams


def get_long_bars(diagrams, dim=0, top_n=10):
    """提取前 top_n 个最长 bar"""
    diag = diagrams[0]
    mask = diag[:, 2] == dim
    bars = diag[mask][:, :2]
    life = bars[:, 1] - bars[:, 0]
    order = np.argsort(-life)[:top_n]
    return pd.DataFrame({
        'birth': bars[order, 0], 'death': bars[order, 1], 'lifetime': life[order]
    })


# ============================================================
# Barcode 图
# ============================================================
def plot_barcode(diagrams, save_path=None):
    """H0 + H1 barcode 并排"""
    diag = diagrams[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for di, dim in enumerate([0, 1]):
        ax = axes[di]
        mask = diag[:, 2] == dim
        bars = diag[mask][:, :2]
        life = bars[:, 1] - bars[:, 0]
        bars = bars[np.argsort(-life)]
        for i, (b, d) in enumerate(bars[:50]):
            c = 'steelblue' if dim == 0 else 'coral'
            lw = 2.5 if (d - b) > np.percentile(life, 90) else 1.0
            ax.plot([b, d], [i, i], color=c, linewidth=lw, alpha=0.8)
        ax.set_xlabel('Filtration')
        ax.set_ylabel('Feature')
        ax.set_title(f'H{dim} Barcode ({mask.sum()} bars)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================
# 聚类数定量确定 (3种方法 + 共识)
# ============================================================
def determine_cluster_number(diagrams, save_path=None):
    """Gap ratio / Threshold / Elbow → Consensus"""
    diag = diagrams[0]
    h0 = diag[diag[:, 2] == 0]
    life = np.sort(h0[:, 1] - h0[:, 0])[::-1]

    n = min(20, len(life))
    ratios = life[:n-1] / (life[1:n] + 1e-10)
    n_gap = int(np.argmax(ratios) + 1)
    n_thresh = int((life > np.median(life) * 2).sum())
    n_elbow = int(np.argmax(np.abs(np.diff(np.diff(life[:n])))) + 2) if n > 3 else n
    consensus = int(np.median([n_gap, n_thresh, n_elbow]))

    print(f"\nLifetimes (top 10): {life[:10].round(2)}")
    print(f"Gap ratios (top 10): {ratios[:10].round(2)}")
    print(f"\n  Gap ratio → k = {n_gap}")
    print(f"  Threshold → k = {n_thresh}")
    print(f"  Elbow     → k = {n_elbow}")
    print(f"  Consensus → k = {consensus}")

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # Barcode (horizontal)
        for i, lt in enumerate(life[:15]):
            axes[0].barh(i, lt, color='#e74c3c' if i < consensus else '#95a5a6',
                         height=0.7, alpha=0.8)
        axes[0].set_ylabel('Index'); axes[0].set_xlabel('Lifetime')
        axes[0].set_title(f'H0 Barcode (red = top {consensus})')
        axes[0].invert_yaxis()
        # Lifetime curve
        axes[1].plot(range(1, n+1), life[:n], 'o-', color='steelblue', lw=2)
        axes[1].axvline(consensus, color='red', ls='--', label=f'k={consensus}')
        axes[1].set_xlabel('Index'); axes[1].set_ylabel('Lifetime')
        axes[1].set_title('Lifetime Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)
        # Gap ratio
        axes[2].bar(range(1, len(ratios[:15])+1), ratios[:15], color='coral', alpha=0.8)
        axes[2].axvline(n_gap, color='red', ls='--', label=f'k={n_gap}')
        axes[2].set_xlabel('Position'); axes[2].set_ylabel('Ratio')
        axes[2].set_title('Gap Ratio'); axes[2].legend(); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    return consensus


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("持久同调分析 + 聚类数估计 (→ 4.1)")
    print("=" * 60)

    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

    # 持久同调
    diagrams = compute_persistence(adata)

    print("\n=== H0 Long Bars ===")
    print(get_long_bars(diagrams, dim=0, top_n=5))
    print("\n=== H1 Long Bars ===")
    print(get_long_bars(diagrams, dim=1, top_n=5))

    # Barcode
    plot_barcode(diagrams, save_path=figp("barcode.png"))
    print(f"\n[Info] Saved: {figp('barcode.png')}")

    # 聚类数
    k = determine_cluster_number(diagrams, save_path=figp("cluster_number.png"))
    print(f"\n[Info] Saved: {figp('cluster_number.png')}")
    print(f"\n[Done] Consensus k = {k}, Mapper comps = {len(comps)}")
