"""
exp_supplement.py — 补充图生成
================================
审查发现的 7 张缺失图, 按优先级生成:

P0 (必须有):
  1. fig/supplement/pipeline_overview.png       整体流程图
  2. fig/supplement/gt_vs_mapper_umap.png       Ground Truth vs Mapper UMAP 并排
  3. fig/supplement/ablation_barcode_compare.png 三种插值 barcode 并排

P1 (强烈建议):
  4. fig/supplement/rarity_framework_2x2.png    丰度×拓扑 2×2 概念框架图
  5. fig/supplement/imbal_case_umap.png          不平衡代表性案例 UMAP

P2 (加分项):
  6. fig/supplement/persistence_diagram.png      Birth-Death 散点图
  7. fig/supplement/mapper_rarity_colored.png    Mapper 图按 rarity score 着色
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import magic
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
from matplotlib.lines import Line2D
from scipy.sparse import issparse
from sklearn.impute import KNNImputer
from sklearn.metrics import adjusted_rand_score
from gtda.homology import VietorisRipsPersistence
import warnings
warnings.filterwarnings('ignore')

from core import (RANDOM_SEED, DATA_PATH, PROJECT_ROOT, FIG_DIR,
                  fig_path, make_fig, ensure_fig_dir,
                  load_and_normalize, full_pipeline,
                  auto_eps, build_mapper, assign_mapper_labels)

figp = make_fig('supplement')
ensure_fig_dir('supplement')
sc.settings.verbosity = 0
np.random.seed(RANDOM_SEED)
#设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ============================================================
# P0-1: 流程图 (Pipeline Overview)
# ============================================================
def plot_pipeline_overview():
    """用 matplotlib 绘制方法流程图"""
    print("\n[1/7] Pipeline Overview")

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18); ax.set_ylim(0, 6)
    ax.axis('off')

    # 定义流程块
    boxes = [
        (1.0, 3.0, 'scRNA-seq\n原始数据\n(1000×23433)', '#f0f0f0'),
        (3.5, 3.0, 'QC\n归一化\nHVG筛选', '#d5e8d4'),
        (6.0, 3.0, 'MAGIC\n扩散插值\n(t=3)', '#dae8fc'),
        (8.5, 3.0, 'PCA\n(30 PCs)', '#fff2cc'),
        (11.0, 4.2, '持久同调\nH0/H1\n聚类数估计', '#f8cecc'),
        (11.0, 1.8, 'Mapper\n1D UMAP lens\nDBSCAN局部聚类', '#e1d5e7'),
        (14.0, 4.2, 'Barcode\n聚类数 k', '#f8cecc'),
        (14.0, 1.8, '拓扑图\n连通分量\n稀有评分', '#e1d5e7'),
        (16.5, 3.0, 'Ground Truth\n对标\nARI/NMI', '#ffe6cc'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x - 0.9, y - 0.65), 1.8, 1.3,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor='#333333',
            linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                fontweight='bold', linespacing=1.3)

    # 箭头
    arrows = [
        (1.9, 3.0, 2.6, 3.0),   # data → QC
        (4.4, 3.0, 5.1, 3.0),   # QC → MAGIC
        (6.9, 3.0, 7.6, 3.0),   # MAGIC → PCA
        (9.4, 3.4, 10.1, 4.0),  # PCA → PH
        (9.4, 2.6, 10.1, 2.0),  # PCA → Mapper
        (11.9, 4.2, 13.1, 4.2), # PH → Barcode
        (11.9, 1.8, 13.1, 1.8), # Mapper → Topo
        (14.9, 4.0, 15.6, 3.4), # Barcode → Benchmark
        (14.9, 2.1, 15.6, 2.7), # Topo → Benchmark
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # 分支标注
    ax.text(9.7, 3.9, '分支1', fontsize=7, color='#b85450', style='italic')
    ax.text(9.7, 2.2, '分支2', fontsize=7, color='#9673a6', style='italic')

    ax.set_title('TDA 单细胞聚类流程总览', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(figp('pipeline_overview.png'), dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: {figp('pipeline_overview.png')}")


# ============================================================
# P0-2: Ground Truth vs Mapper UMAP 并排
# ============================================================
def plot_gt_vs_mapper_umap(adata, mapper_labels):
    """2 面板: (A) UMAP 按 10 种真实类型着色 (B) UMAP 按 5 个 Mapper 分量着色"""
    print("\n[2/7] Ground Truth vs Mapper UMAP")

    gt = pd.read_csv(os.path.join(PROJECT_ROOT, 'cell_type_labels.csv')).set_index('cell_id')
    common = adata.obs_names.intersection(gt.index)
    idx = [list(adata.obs_names).index(c) for c in common]
    gt_aligned = gt.loc[common, 'cell_type'].values
    ml = mapper_labels[idx]
    umap = adata.obsm['X_umap'][idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # (A) Ground Truth
    ax = axes[0]
    types = sorted(set(gt_aligned))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(types)))
    type_colors = {t: cmap[i] for i, t in enumerate(types)}
    for t in types:
        mask = gt_aligned == t
        ax.scatter(umap[mask, 0], umap[mask, 1], c=[type_colors[t]], s=8,
                   alpha=0.7, label=t[:20])
    ax.set_title(f'(A) Ground Truth ({len(types)} cell types)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=7, markerscale=2, loc='best', ncol=1,
              framealpha=0.9, title='Cell Type', title_fontsize=8)

    # (B) Mapper
    ax = axes[1]
    comp_ids = sorted(set(ml[ml >= 0]))
    cmap_m = plt.cm.Set1(np.linspace(0, 1, max(len(comp_ids), 1)))
    for ci, comp in enumerate(comp_ids):
        mask = ml == comp
        n = mask.sum()
        # 找主导类型
        dominant = pd.Series(gt_aligned[mask]).value_counts().index[0]
        ax.scatter(umap[mask, 0], umap[mask, 1], c=[cmap_m[ci]], s=8, alpha=0.7,
                   label=f'Comp_{comp} (n={n}, {dominant[:15]})')
    unassigned = ml < 0
    if unassigned.sum() > 0:
        ax.scatter(umap[unassigned, 0], umap[unassigned, 1], c='lightgray',
                   s=4, alpha=0.3, label='Unassigned')

    ari = adjusted_rand_score(gt_aligned, ml)
    ax.set_title(f'(B) Mapper ({len(comp_ids)} components, ARI={ari:.3f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.legend(fontsize=7, markerscale=2, loc='best', framealpha=0.9,
              title='Mapper Component', title_fontsize=8)

    plt.tight_layout()
    plt.savefig(figp('gt_vs_mapper_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('gt_vs_mapper_umap.png')}")


# ============================================================
# P0-3: 三种插值 Barcode 并排
# ============================================================
def plot_ablation_barcodes():
    """3 面板: 无插值 / KNN / MAGIC 的 H0 barcode 并排"""
    print("\n[3/7] Ablation Barcode Comparison")

    base = load_and_normalize(DATA_PATH)

    conditions = {}

    # (A) 无插值
    a = base.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor='seurat', subset=True)
    sc.tl.pca(a, n_comps=30, random_state=RANDOM_SEED)
    conditions['No Imputation'] = a

    # (B) KNN
    b = base.copy()
    Xb = b.X.toarray() if issparse(b.X) else b.X.copy()
    Xb[Xb == 0] = np.nan
    Xb = np.maximum(KNNImputer(n_neighbors=10).fit_transform(Xb), 0)
    b.X = Xb
    sc.pp.highly_variable_genes(b, n_top_genes=2000, flavor='seurat', subset=True)
    sc.tl.pca(b, n_comps=30, random_state=RANDOM_SEED)
    conditions['KNN Imputation'] = b

    # (C) MAGIC
    c = base.copy()
    if issparse(c.X): c.X = np.expm1(c.X.toarray())
    else: c.X = np.expm1(c.X)
    c.X = np.sqrt(c.X)
    c.X = magic.MAGIC(t=3, n_pca=20, n_jobs=-1, random_state=RANDOM_SEED,
                       verbose=0).fit_transform(c.to_df()).values
    sc.pp.highly_variable_genes(c, n_top_genes=2000, flavor='seurat', subset=True)
    sc.tl.pca(c, n_comps=30, random_state=RANDOM_SEED)
    conditions['MAGIC (t=3)'] = c

    # 计算持久同调并绘图
    vr = VietorisRipsPersistence(homology_dimensions=[0], n_jobs=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71']

    for ax, (name, ad), color in zip(axes, conditions.items(), colors_bar):
        diag = vr.fit_transform(ad.obsm['X_pca'][np.newaxis, :, :])[0]
        h0 = diag[diag[:, 2] == 0]
        life = h0[:, 1] - h0[:, 0]
        life_sorted = np.sort(life)[::-1]

        n_sig = int((life > np.median(life) * 3).sum())

        # 画 barcode
        for i, lt in enumerate(life_sorted[:40]):
            is_sig = lt > np.median(life) * 3
            lw = 2.5 if is_sig else 0.8
            alpha = 0.9 if is_sig else 0.4
            ax.barh(i, lt, color=color, height=0.7, alpha=alpha, linewidth=0)

        ax.set_xlabel('Lifetime', fontsize=11)
        ax.set_ylabel('Feature index', fontsize=11)
        ax.set_title(f'{name}\nH0 significant bars: {n_sig}',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 标注
        if n_sig == 0:
            ax.text(0.5, 0.5, 'NO SIGNIFICANT\nH0 BARS',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=16, fontweight='bold', color='red', alpha=0.6)

    plt.suptitle('H0 Persistence Barcode: Imputation Method Comparison\n'
                 'KNN destroys all topological structure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('ablation_barcode_compare.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('ablation_barcode_compare.png')}")


# ============================================================
# P1-4: 丰度×拓扑 2×2 概念框架图
# ============================================================
def plot_rarity_framework():
    """2×2 概念框架: 丰度稀有性 × 拓扑可分离性"""
    print("\n[4/7] Rarity Framework 2×2")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    # 四个象限
    quads = [
        # (x, y, 标题, 案例, 颜色, Mapper结果)
        (0.5, 1.75,
         '丰度稀有 + 拓扑可分离',
         'enterocyte n=20\nkeratinocyte n=20',
         '#2ecc71', 'Mapper F1=0.83-0.97\n 独立连通分量'),
        (1.75, 1.75,
         '丰度稀有 + 拓扑不可分离',
         'B cell n=50\nimmature T cell n=50',
         '#e74c3c', 'Mapper F1=0\n 被吸收进Comp_0'),
        (0.5, 0.55,
         '丰度不稀有 + 拓扑可分离',
         'enterocyte (10%, 均衡)\nbasal cell (10%, 均衡)',
         '#3498db', 'Mapper 纯度93-100%\n 独立连通分量'),
        (1.75, 0.55,
         '丰度不稀有 + 拓扑不可分离',
         'B cell (10%, 均衡)\nfibroblast (10%, 均衡)',
         '#95a5a6', '全部合并到Comp_0\n17% each in 583 cells'),
    ]

    for x, y, title, case, color, result in quads:
        # 背景框
        rect = mpatches.FancyBboxPatch((x - 0.45, y - 0.42), 0.9, 0.84,
            boxstyle="round,pad=0.05", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        # 标题
        ax.text(x, y + 0.28, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        # 案例
        ax.text(x, y + 0.05, case, ha='center', va='center',
                fontsize=8, color='#333333', style='italic', linespacing=1.4)
        # 结果
        ax.text(x, y - 0.22, result, ha='center', va='center',
                fontsize=7.5, color='#555555', linespacing=1.4,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    # 轴标签
    ax.annotate('', xy=(2.3, 1.15), xytext=(-0.1, 1.15),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    ax.text(1.1, 1.08, '拓扑可分离性 →', ha='center', fontsize=11,
            fontweight='bold', color='#333333')
    ax.text(-0.15, 1.75, '高', ha='center', fontsize=9, color='#333333')
    ax.text(2.35, 1.75, '低', ha='center', fontsize=9, color='#333333')

    ax.annotate('', xy=(1.1, 2.35), xytext=(1.1, 0.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    ax.text(1.03, 1.18, '↑\n丰\n度\n稀\n有\n性', ha='center', fontsize=9,
            fontweight='bold', color='#333333', linespacing=0.8)
    ax.text(0.5, 2.25, '稀有 (<5%)', ha='center', fontsize=9, color='#333333')
    ax.text(0.5, 0.05, '不稀有 (≥5%)', ha='center', fontsize=9, color='#333333')

    ax.set_title('丰度稀有性 × 拓扑可分离性: 两个独立维度\n'
                 '(基于均衡实验 + 12组不平衡实验)',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(figp('rarity_framework_2x2.png'), dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: {figp('rarity_framework_2x2.png')}")


# ============================================================
# P1-5: 不平衡代表性案例 UMAP
# ============================================================
def plot_imbal_case_umap():
    """keratinocyte n=20 (成功) vs B cell n=20 (失败) 的 UMAP 对比"""
    print("\n[5/7] Imbalanced Case UMAP")

    IMBAL_DIR = os.path.join(PROJECT_ROOT, 'imbalance_data')
    cases = [
        ('keratinocyte_stem_cell', 20, '高可分离', 'F1=0.97'),
        ('B_cell', 20, '低可分离', 'F1=0'),
    ]

    found_cases = []
    for fname_base, n_rare, topo, f1_label in cases:
        data_f = os.path.join(IMBAL_DIR, f'sce_imbal_{fname_base}_n{n_rare}.csv')
        label_f = os.path.join(IMBAL_DIR, f'labels_imbal_{fname_base}_n{n_rare}.csv')
        if os.path.exists(data_f) and os.path.exists(label_f):
            found_cases.append((data_f, label_f, fname_base, n_rare, topo, f1_label))

    if len(found_cases) < 2:
        print("  ⚠ 不平衡数据不完整，跳过")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    for ax, (data_f, label_f, fname, n_rare, topo, f1_label) in zip(axes, found_cases):
        # 加载
        adata = sc.read_csv(data_f)
        gt = pd.read_csv(label_f)
        if 'Unnamed: 0' in gt.columns: gt = gt.drop(columns='Unnamed: 0')
        gt = gt.set_index('cell_id')

        # 预处理 (简化版)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', subset=True)
        sc.tl.pca(adata, n_comps=30, random_state=RANDOM_SEED)
        sc.pp.neighbors(adata, random_state=RANDOM_SEED)
        sc.tl.umap(adata, random_state=RANDOM_SEED)

        # 对齐
        common = adata.obs_names.intersection(gt.index)
        idx = [list(adata.obs_names).index(c) for c in common]
        gt_al = gt.loc[common, 'cell_type'].values
        umap = adata.obsm['X_umap'][idx]

        rare_type = fname.replace('_', ' ')
        is_rare = np.array([t == rare_type for t in gt_al])

        # 画图: 灰色=主群体, 红色=稀有
        ax.scatter(umap[~is_rare, 0], umap[~is_rare, 1],
                   c='lightgray', s=6, alpha=0.5, label='Major types')
        ax.scatter(umap[is_rare, 0], umap[is_rare, 1],
                   c='red', s=30, alpha=0.9, edgecolors='darkred', linewidths=0.5,
                   label=f'{rare_type} (n={is_rare.sum()})')
        ax.set_title(f'{rare_type[:25]} n={n_rare}\n{topo}, Mapper {f1_label}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, markerscale=1.5)
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

    plt.suptitle('Imbalanced Experiment: Success vs Failure\n'
                 '(Red = rare cells, Gray = major types)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('imbal_case_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('imbal_case_umap.png')}")


# ============================================================
# P2-6: Persistence Diagram 散点图
# ============================================================
def plot_persistence_diagram(adata):
    """经典 birth-death 散点图 (H0 + H1)"""
    print("\n[6/7] Persistence Diagram")

    vr = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
    diag = vr.fit_transform(adata.obsm['X_pca'][np.newaxis, :, :])[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for di, (dim, color, name) in enumerate([(0, 'steelblue', 'H0 (Components)'),
                                               (1, 'coral', 'H1 (Loops)')]):
        ax = axes[di]
        mask = diag[:, 2] == dim
        births = diag[mask, 0]
        deaths = diag[mask, 1]
        life = deaths - births

        # 对角线
        lim = max(deaths.max(), births.max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Diagonal (b=d)')

        # 按 lifetime 着色
        sc_plot = ax.scatter(births, deaths, c=life, cmap='YlOrRd', s=30,
                              alpha=0.8, edgecolors='white', linewidths=0.3)

        # 标注长寿命特征
        thresh = np.median(life) * 3
        sig_mask = life > thresh
        if sig_mask.sum() > 0:
            ax.scatter(births[sig_mask], deaths[sig_mask],
                       facecolors='none', edgecolors=color, s=100, linewidths=2,
                       label=f'Significant ({sig_mask.sum()} bars)')

        plt.colorbar(sc_plot, ax=ax, label='Lifetime', fraction=0.046, pad=0.04)
        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(f'{name}\n{mask.sum()} features, {sig_mask.sum()} significant',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)

    plt.suptitle('Persistence Diagram (MAGIC t=3, PCA 30D)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('persistence_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('persistence_diagram.png')}")


# ============================================================
# P2-7: Mapper 图按 Rarity Score 着色
# ============================================================
def plot_mapper_rarity_colored(graph, G, adata):
    """Mapper 图节点颜色 = rarity score 热力图"""
    print("\n[7/7] Mapper Rarity Colored")

    # 计算 rarity score (复用 exp_rarity 的逻辑)
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

    rarity_scores = {}
    node_sizes = {}
    node_roles = {}

    for n in G.nodes():
        mem = graph['nodes'][n]
        d = G.degree(n)
        if d == 0: role = 'Isolated'
        elif d == 1: role = 'Terminal'
        elif n in artic: role = 'Bridge'
        elif d >= 3: role = 'Hub'
        else: role = 'Chain'
        node_roles[n] = role
        node_sizes[n] = len(mem)

        md = min((nx.shortest_path_length(G, n, h)
                  for h in hubs if nx.has_path(G, n, h)), default=999)
        nbs = [len(graph['nodes'][nb]) for nb in G.neighbors(n)]
        dr = len(mem) / (np.mean(nbs) + 1e-6) if nbs else 0.5
        td = np.linalg.norm(np.mean(X[mem], axis=0) - gm)
        rarity_scores[n] = {'topo_iso': md, 'density_ratio': dr,
                              'trans_dist': td, 'n_cells': len(mem)}

    # 归一化
    df = pd.DataFrame(rarity_scores).T
    for c in ['topo_iso', 'trans_dist']:
        mn, mx = df[c].min(), df[c].max()
        df[f'{c}_n'] = (df[c] - mn) / (mx - mn + 1e-8)
    mn, mx = df['density_ratio'].min(), df['density_ratio'].max()
    df['density_n'] = 1.0 - (df['density_ratio'] - mn) / (mx - mn + 1e-8)
    median_size = df['n_cells'].median()
    df['size_penalty'] = np.clip(df['n_cells'] / (median_size + 1e-12), 0, 1)
    df['rarity'] = (0.3 * df['topo_iso_n'] + 0.2 * df['density_n'] +
                     0.2 * df['trans_dist_n'] + 0.3 * (1.0 - df['size_penalty']))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=1.5, iterations=100)

    # (A) 按 rarity score 着色
    ax = axes[0]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)
    nodes = list(G.nodes())
    rarity_vals = [df.loc[n, 'rarity'] for n in nodes]
    sizes = [node_sizes[n] * 5 + 40 for n in nodes]
    sc_plot = nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
        node_color=rarity_vals, cmap='YlOrRd', node_size=sizes,
        alpha=0.9, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc_plot, ax=ax, label='Rarity Score', fraction=0.046, pad=0.04)
    # 标注高 rarity 节点
    for n in nodes:
        if df.loc[n, 'rarity'] > 0.7:
            x, y = pos[n]
            ax.annotate(f'{df.loc[n, "rarity"]:.2f}\n({node_sizes[n]}c)',
                        (x, y), fontsize=7, ha='center', va='bottom',
                        fontweight='bold', color='darkred')
    ax.set_title('(A) Mapper Graph: Rarity Score', fontsize=13, fontweight='bold')
    ax.axis('off')

    # (B) 按节点角色着色 (对比参考)
    ax = axes[1]
    role_colors = {'Terminal': '#e74c3c', 'Bridge': '#f39c12', 'Hub': '#2ecc71',
                   'Chain': '#95a5a6', 'Isolated': '#8e44ad'}
    colors = [role_colors.get(node_roles[n], '#95a5a6') for n in nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
        node_color=colors, node_size=sizes, alpha=0.9,
        edgecolors='white', linewidths=0.5)
    for n in nodes:
        x, y = pos[n]
        if node_sizes[n] >= 30 or G.degree(n) >= 3:
            ax.annotate(f'{node_sizes[n]}', (x, y), fontsize=7,
                        ha='center', va='center', fontweight='bold')
    ax.legend(handles=[Line2D([0], [0], marker='o', color='w',
              markerfacecolor=c, markersize=10, label=r)
              for r, c in role_colors.items()], loc='upper left', fontsize=9)
    ax.set_title('(B) Mapper Graph: Node Roles', fontsize=13, fontweight='bold')
    ax.axis('off')

    plt.suptitle('Mapper Graph: Rarity Score vs Node Roles\n'
                 '(size ∝ cells, Terminal+small = high rarity)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figp('mapper_rarity_colored.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figp('mapper_rarity_colored.png')}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("补充图生成 (7张)")
    print("=" * 65)

    # P0-1: 流程图 (不需要数据)
    plot_pipeline_overview()

    # 加载数据 (用于后续所有图)
    adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

    # P0-2: GT vs Mapper UMAP
    plot_gt_vs_mapper_umap(adata, mapper_labels)

    # P0-3: 三种插值 barcode
    plot_ablation_barcodes()

    # P1-4: 2×2 概念框架
    plot_rarity_framework()

    # P1-5: 不平衡案例 UMAP
    plot_imbal_case_umap()

    # P2-6: Persistence diagram
    plot_persistence_diagram(adata)

    # P2-7: Mapper rarity
    plot_mapper_rarity_colored(graph, G, adata)

    print(f"\n{'=' * 65}")
    print("全部 7 张补充图已保存到 fig/supplement/")
    print("=" * 65)
    print("""
  fig/supplement/
  ├── pipeline_overview.png          P0 流程图
  ├── gt_vs_mapper_umap.png          P0 GT vs Mapper UMAP
  ├── ablation_barcode_compare.png   P0 三种插值 barcode
  ├── rarity_framework_2x2.png       P1 2×2 概念框架
  ├── imbal_case_umap.png            P1 不平衡案例对比
  ├── persistence_diagram.png        P2 Birth-Death 散点
  └── mapper_rarity_colored.png      P2 Rarity Score 着色
    """)
    print("[Done]")
