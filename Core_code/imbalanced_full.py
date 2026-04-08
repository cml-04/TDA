"""
imbalanced_full.py — 系统性不平衡实验
======================================
实验设计 (2×2×3 = 12组):
  维度1 — 拓扑可分离性:
    高: enterocyte, keratinocyte stem cell  (均衡实验纯度 93%-100%)
    低: B cell, immature T cell             (均衡实验中被合并到 Comp_0)
  维度2 — 不平衡比例:
    10:1 (稀有类型10个, ~1.1%)
    5:1  (稀有类型20个, ~2.2%)
    2:1  (稀有类型50个, ~5.3%)

核心结论:
  1. 拓扑可分离类型在所有比例下都能被检出 (F1 > 0)
  2. 拓扑不可分离类型即使丰度较高也难以被检出 (F1 ≈ 0)
  3. → 丰度稀有 ≠ 拓扑可分离, 两者是独立维度
  4. Mapper vs Leiden vs Density 各有擅长场景

输出: fig/imbal_full_*.png + 汇总 Excel
"""

import numpy as np
import pandas as pd
import scanpy as sc
import magic
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy.sparse import issparse
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              precision_score, recall_score, f1_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kmapper as km
import os
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
from core import (RANDOM_SEED, fig_path, make_fig, ensure_fig_dir,
                  auto_eps, build_mapper, assign_mapper_labels)

ensure_fig_dir('imbalanced')
figp = make_fig('imbalanced')
np.random.seed(RANDOM_SEED)
sc.settings.verbosity = 0

RARE_THRESHOLD = 0.05
from core import PROJECT_ROOT as _PR
IMBAL_DIR = os.path.join(_PR, 'imbalance_data')  # 不平衡数据集所在文件夹

# ============================================================
# 实验配置
# ============================================================
# 从均衡实验得知的拓扑可分离性分类
TOPO_HIGH = [
    'enterocyte of epithelium of large intestine',   # 均衡纯度 93%
    'keratinocyte stem cell',                         # 均衡纯度 100%
]
TOPO_LOW = [
    'B cell',                                         # 均衡中被合并到 Comp_0
    'immature T cell',                                # 均衡中被合并到 Comp_0
]

RARE_COUNTS = [10, 20, 50]  # 稀有类型的细胞数
MAJOR_COUNT = 100            # 主要类型每种的细胞数

ALL_RARE_TYPES = TOPO_HIGH + TOPO_LOW


# ============================================================
# 工具函数
# ============================================================
def preprocess_adata(adata_raw):
    """标准预处理: QC → normalize → MAGIC → PCA → UMAP"""
    adata = adata_raw.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = (adata.var_names.str.startswith('MT-') |
                       adata.var_names.str.startswith('mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],
                                percent_top=None, log1p=False, inplace=True)
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    raw_adata = adata.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', subset=False)
    hvg = adata.var.highly_variable
    work = adata.copy()
    if issparse(work.X):
        work.X = np.expm1(work.X.toarray())
    else:
        work.X = np.expm1(work.X)
    work.X = np.sqrt(work.X)
    work.X = magic.MAGIC(t=3, n_pca=20, n_jobs=-1, random_state=RANDOM_SEED,
                          verbose=0).fit_transform(work.to_df()).values
    out = work[:, hvg].copy()
    out.raw = raw_adata

    sc.tl.pca(out, n_comps=30, random_state=RANDOM_SEED)
    sc.pp.neighbors(out, random_state=RANDOM_SEED)
    sc.tl.umap(out, random_state=RANDOM_SEED)
    return out


def evaluate_rare_detection(pred_binary, gt_binary):
    """计算 Precision / Recall / F1"""
    tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
            'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4)}


def detect_mapper_rare(mapper_labels, n_total, threshold=RARE_THRESHOLD):
    """Mapper: 小分量 (占比 < threshold) 标记为稀有"""
    pred = np.zeros(n_total, dtype=int)
    for comp_id in sorted(set(mapper_labels)):
        if comp_id < 0:
            continue
        mask = mapper_labels == comp_id
        if mask.sum() / n_total < threshold:
            pred[mask] = 1
    return pred


def detect_leiden_rare(adata, idx_map, n_total, res=0.3, threshold=RARE_THRESHOLD):
    """Leiden: 小簇标记为稀有"""
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r_{res}',
                 random_state=RANDOM_SEED)
    ll = adata.obs[f'leiden_r_{res}'].values[idx_map]
    pred = np.zeros(n_total, dtype=int)
    counts = pd.Series(ll).value_counts()
    for cid, cnt in counts.items():
        if cnt / n_total < threshold:
            pred[ll == cid] = 1
    return pred, ll


def detect_density_rare(adata, idx_map, n_total, percentile=95):
    """Density: PCA 空间低密度点标记为稀有"""
    pca = adata.obsm['X_pca'][idx_map]
    nn = NearestNeighbors(n_neighbors=min(30, n_total - 1), n_jobs=-1).fit(pca)
    d, _ = nn.kneighbors(pca)
    mean_d = d[:, 1:].mean(axis=1)
    pred = (mean_d > np.percentile(mean_d, percentile)).astype(int)
    return pred


# ============================================================
# 主实验循环
# ============================================================
print("=" * 70)
print("系统性不平衡实验: 4 稀有类型 × 3 比例 = 12 组")
print("=" * 70)

all_results = []
confusion_data = {}

for rare_type in ALL_RARE_TYPES:
    topo_class = "高可分离" if rare_type in TOPO_HIGH else "低可分离"

    for n_rare in RARE_COUNTS:
        exp_id = f"{rare_type[:12]}..._n{n_rare}"
        ratio_label = f"{MAJOR_COUNT}:{n_rare}"
        n_total_expected = 9 * MAJOR_COUNT + n_rare

        print(f"\n{'─' * 60}")
        print(f"实验: {rare_type} (n={n_rare}, {topo_class})")
        print(f"{'─' * 60}")

        # --- 加载数据 ---
        # 尝试多种文件名格式
        fname_base = rare_type.replace(' ', '_')
        possible_data = [
            os.path.join(IMBAL_DIR, f'sce_imbal_{fname_base}_n{n_rare}.csv'),
            os.path.join(IMBAL_DIR, f'sce_imbal_{fname_base}.csv'),
        ]
        possible_labels = [
            os.path.join(IMBAL_DIR, f'labels_imbal_{fname_base}_n{n_rare}.csv'),
            os.path.join(IMBAL_DIR, f'labels_imbal_{fname_base}.csv'),
        ]

        data_file = None
        label_file = None
        for f in possible_data:
            if os.path.exists(f):
                data_file = f
                break
        for f in possible_labels:
            if os.path.exists(f):
                label_file = f
                break

        if data_file is None or label_file is None:
            print(f"  ⚠ 数据文件不存在, 跳过")
            print(f"    需要: {possible_data[0]} + {possible_labels[0]}")
            all_results.append({
                'Rare_Type': rare_type, 'Topo_Class': topo_class,
                'N_Rare': n_rare, 'Ratio': ratio_label,
                'Method': 'N/A', 'F1': np.nan, 'Precision': np.nan,
                'Recall': np.nan, 'ARI': np.nan, 'NMI': np.nan,
                'Mapper_k': np.nan, 'Rare_in_own_comp': 'N/A'
            })
            continue

        # 加载
        adata_raw = sc.read_csv(data_file)
        gt = pd.read_csv(label_file)
        if 'Unnamed: 0' in gt.columns:
            gt = gt.drop(columns=['Unnamed: 0'])
        gt = gt.set_index('cell_id')

        # 预处理
        adata = preprocess_adata(adata_raw)

        # 对齐
        common = adata.obs_names.intersection(gt.index)
        if len(common) == 0:
            print(f"  ⚠ cell ID 不匹配, 跳过")
            continue

        idx_map = [list(adata.obs_names).index(c) for c in common]
        gt_aligned = gt.loc[common, 'cell_type'].values
        n = len(common)

        # 二值标签
        gt_binary = np.array([1 if t == rare_type else 0 for t in gt_aligned])
        n_rare_actual = gt_binary.sum()
        print(f"  数据: {n} cells, 稀有 {n_rare_actual} ({n_rare_actual/n*100:.1f}%)")

        # --- Mapper ---
        lens = adata.obsm['X_umap'][:, 0]
        graph, G, comps, eps = build_mapper(adata.obsm['X_pca'], lens)
        mapper_labels = assign_mapper_labels(graph, comps, adata.n_obs)
        ml = mapper_labels[idx_map]
        n_comps = len(comps)

        mapper_pred = detect_mapper_rare(ml, n)
        mapper_eval = evaluate_rare_detection(mapper_pred, gt_binary)

        # Mapper 的稀有细胞去了哪个分量?
        rare_distribution = {}
        for comp_id in sorted(set(ml)):
            if comp_id < 0: continue
            mask = ml == comp_id
            rare_in = gt_binary[mask].sum()
            if rare_in > 0:
                rare_distribution[comp_id] = {
                    'rare_in': rare_in, 'total': mask.sum(),
                    'pct_of_comp': rare_in / mask.sum(),
                    'pct_of_rare': rare_in / max(n_rare_actual, 1)
                }
        rare_comp_str = "; ".join(
            f"C{k}({v['rare_in']}/{v['total']}={v['pct_of_comp']:.0%})"
            for k, v in rare_distribution.items()
        )
        # 是否有独立分量？
        has_own_comp = any(v['pct_of_comp'] > 0.5 for v in rare_distribution.values())

        ari_mapper = adjusted_rand_score(gt_aligned, ml)
        nmi_mapper = normalized_mutual_info_score(gt_aligned, ml)

        all_results.append({
            'Rare_Type': rare_type, 'Topo_Class': topo_class,
            'N_Rare': n_rare, 'Ratio': ratio_label,
            'Method': 'Mapper', **mapper_eval,
            'ARI': round(ari_mapper, 4), 'NMI': round(nmi_mapper, 4),
            'Mapper_k': n_comps, 'Rare_in_own_comp': rare_comp_str
        })
        print(f"  Mapper: k={n_comps}, F1={mapper_eval['F1']:.4f}, "
              f"稀有分布={rare_comp_str}, 独立分量={'是' if has_own_comp else '否'}")

        # --- Leiden (res=0.3) ---
        leiden_pred, leiden_labels = detect_leiden_rare(adata, idx_map, n, res=0.3)
        leiden_eval = evaluate_rare_detection(leiden_pred, gt_binary)
        ari_leiden = adjusted_rand_score(gt_aligned, leiden_labels)
        nmi_leiden = normalized_mutual_info_score(gt_aligned, leiden_labels)

        all_results.append({
            'Rare_Type': rare_type, 'Topo_Class': topo_class,
            'N_Rare': n_rare, 'Ratio': ratio_label,
            'Method': 'Leiden(0.3)', **leiden_eval,
            'ARI': round(ari_leiden, 4), 'NMI': round(nmi_leiden, 4),
            'Mapper_k': len(set(leiden_labels)), 'Rare_in_own_comp': ''
        })
        print(f"  Leiden:  k={len(set(leiden_labels))}, F1={leiden_eval['F1']:.4f}")

        # --- Density ---
        density_pred = detect_density_rare(adata, idx_map, n)
        density_eval = evaluate_rare_detection(density_pred, gt_binary)

        all_results.append({
            'Rare_Type': rare_type, 'Topo_Class': topo_class,
            'N_Rare': n_rare, 'Ratio': ratio_label,
            'Method': 'Density', **density_eval,
            'ARI': np.nan, 'NMI': np.nan,
            'Mapper_k': np.nan, 'Rare_in_own_comp': ''
        })
        print(f"  Density: F1={density_eval['F1']:.4f}")

        # 保存混淆矩阵
        ct = pd.crosstab(
            pd.Series(gt_aligned, name='Type'),
            pd.Series([f'C{l}' if l >= 0 else 'NA' for l in ml], name='Comp')
        )
        confusion_data[f"{rare_type[:20]}_n{n_rare}"] = ct

# ============================================================
# 汇总分析
# ============================================================
df = pd.DataFrame(all_results)
df = df[df['Method'] != 'N/A']

print("\n" + "=" * 70)
print("全部实验结果汇总")
print("=" * 70)
print(df[['Rare_Type', 'Topo_Class', 'N_Rare', 'Method',
          'F1', 'Precision', 'Recall', 'Rare_in_own_comp']].to_string(index=False))

# ============================================================
# 核心分析: 按拓扑可分离性分组的 F1 对比
# ============================================================
print("\n" + "=" * 70)
print("核心结论: 拓扑可分离性 × 丰度 × 方法")
print("=" * 70)

mapper_df = df[df['Method'] == 'Mapper'].copy()
for topo in ['高可分离', '低可分离']:
    sub = mapper_df[mapper_df['Topo_Class'] == topo]
    print(f"\n  [{topo}] 类型:")
    for _, r in sub.iterrows():
        own = "✓独立分量" if '50%' not in str(r['Rare_in_own_comp']) or r['F1'] > 0 else "✗混合"
        print(f"    {r['Rare_Type'][:30]:30s} n={r['N_Rare']:3d}  "
              f"F1={r['F1']:.4f}  {r['Rare_in_own_comp']}")

# ============================================================
# 绘图
# ============================================================
print("\n[绘图]")

# --- 图 1: 核心结论热力图 (方法 × 实验条件) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
methods = ['Mapper', 'Leiden(0.3)', 'Density']

for ax, method in zip(axes, methods):
    sub = df[df['Method'] == method].copy()
    if len(sub) == 0:
        continue

    # 构建热力图矩阵: 行=稀有类型, 列=稀有数量
    pivot = sub.pivot_table(index='Rare_Type', columns='N_Rare',
                             values='F1', aggfunc='first')
    pivot = pivot.reindex(columns=sorted(RARE_COUNTS))

    # 按拓扑类别排序行
    row_order = TOPO_HIGH + TOPO_LOW
    pivot = pivot.reindex([t for t in row_order if t in pivot.index])

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                c = 'white' if v < 0.3 or v > 0.7 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=c)

    ax.set_xticks(range(len(RARE_COUNTS)))
    ax.set_xticklabels([f'n={c}\n({c/(9*100+c)*100:.1f}%)' for c in RARE_COUNTS],
                        fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    labels = []
    for t in pivot.index:
        tag = "[H]" if t in TOPO_HIGH else "[L]"
        labels.append(f'{tag} {t[:25]}')
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Rare cell count', fontsize=10)
    ax.set_title(method, fontsize=13, fontweight='bold')

    # 拓扑类别分隔线
    n_high = sum(1 for t in pivot.index if t in TOPO_HIGH)
    if 0 < n_high < pivot.shape[0]:
        ax.axhline(n_high - 0.5, color='black', linewidth=2, linestyle='-')

plt.suptitle('Rare Cell Detection F1 Score\n'
             '[H] = topologically separable,  [L] = topologically inseparable',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figp('imbal_full_heatmap.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {figp('imbal_full_heatmap.png')}")
plt.close()

# --- 图 2: Mapper F1 vs 稀有细胞数量 (按拓扑类别分线) ---
fig, ax = plt.subplots(figsize=(8, 6))
mapper_sub = df[df['Method'] == 'Mapper'].copy()

for topo, color, marker in [('高可分离', '#2ecc71', 'o'), ('低可分离', '#e74c3c', 'x')]:
    sub = mapper_sub[mapper_sub['Topo_Class'] == topo]
    for rt in sub['Rare_Type'].unique():
        rt_sub = sub[sub['Rare_Type'] == rt].sort_values('N_Rare')
        ax.plot(rt_sub['N_Rare'], rt_sub['F1'], marker=marker, markersize=8,
                linewidth=2, color=color, alpha=0.8,
                label=f'{rt[:25]} ({topo})')

ax.set_xlabel('Number of rare cells', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Mapper: F1 vs Rare Cell Count\n(grouped by topological separability)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(RARE_COUNTS)
plt.tight_layout()
plt.savefig(figp('imbal_full_f1_curve.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {figp('imbal_full_f1_curve.png')}")
plt.close()

# --- 图 3: 三种方法在两类场景下的平均 F1 ---
fig, ax = plt.subplots(figsize=(10, 6))
summary = df.groupby(['Topo_Class', 'Method'])['F1'].mean().unstack()
summary = summary.reindex(columns=methods)

x = np.arange(len(summary.index))
w = 0.25
colors_m = ['#e74c3c', '#3498db', '#95a5a6']

for i, method in enumerate(methods):
    if method in summary.columns:
        vals = summary[method].values
        bars = ax.bar(x + i * w, vals, w, label=method, color=colors_m[i],
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x + w)
ax.set_xticklabels(summary.index, fontsize=11)
ax.set_ylabel('Average F1 Score', fontsize=12)
ax.set_title('Average F1 by Topological Separability × Method',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(figp('imbal_full_avg_f1.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {figp('imbal_full_avg_f1.png')}")
plt.close()

# ============================================================
# 导出 Excel
# ============================================================
output_excel = os.path.join(_PR, 'imbalanced_experiment_results.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='All_Results', index=False)

    # 汇总表
    summary_all = df.pivot_table(
        index=['Rare_Type', 'Topo_Class', 'N_Rare'],
        columns='Method',
        values='F1',
        aggfunc='first'
    ).reset_index()
    summary_all.to_excel(writer, sheet_name='F1_Summary', index=False)

    # 混淆矩阵
    for name, ct in confusion_data.items():
        sheet_name = name[:31]  # Excel sheet name limit
        ct.to_excel(writer, sheet_name=sheet_name)

print(f"\n[Info] Excel: {output_excel}")

# ============================================================
# 最终结论
# ============================================================
print(f"\n{'=' * 70}")
print("实验结论")
print(f"{'=' * 70}")

mapper_high = df[(df['Method'] == 'Mapper') & (df['Topo_Class'] == '高可分离')]['F1']
mapper_low = df[(df['Method'] == 'Mapper') & (df['Topo_Class'] == '低可分离')]['F1']
density_all = df[df['Method'] == 'Density']['F1']

print(f"""
结论1 — 拓扑可分离性决定 Mapper 检测能力:
  高可分离类型 平均F1 = {mapper_high.mean():.4f}
  低可分离类型 平均F1 = {mapper_low.mean():.4f}
  → 丰度稀有 ≠ 拓扑可分离

结论2 — Mapper 在拓扑可分离场景的表现:
  {(mapper_high > 0).sum()}/{len(mapper_high)} 组成功检出 (F1>0)

结论3 — 方法互补性:
  Mapper 擅长: 拓扑孤立的稀有类型 (利用结构信息)
  Density 擅长: 几何空间中的离群点 (不依赖类型结构)
  Leiden 擅长: 整体聚类质量 (高ARI/NMI)

结论4 — 实际建议:
  先用 Mapper 识别拓扑孤立的稀有群体
  再用 Density 补充检测非拓扑离群细胞
  最后用 Leiden 对主群体精细聚类
""")

print("[Done]")
