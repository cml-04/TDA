"""
benchmark.py — Mapper vs Ground Truth 对标分析
==============================================
生成:
  fig/confusion_heatmap.png   Mapper 分量 × 细胞类型 混淆矩阵热力图
  fig/benchmark_summary.png   ARI/NMI/Deviation 综合对比图
  控制台输出 ARI/NMI/Deviation 评级表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import os
from core import full_pipeline, fig_path, make_fig, ensure_fig_dir, RANDOM_SEED, PROJECT_ROOT

ensure_fig_dir('benchmark')
figp = make_fig('benchmark')

# ============================================================
# 1. 运行流程
# ============================================================
adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

# ============================================================
# 2. 加载 Ground Truth
# ============================================================
gt = pd.read_csv(os.path.join(PROJECT_ROOT, 'cell_type_labels.csv'))
gt = gt.set_index('cell_id')

common = adata.obs_names.intersection(gt.index)
print(f"\n匹配到 {len(common)}/{adata.n_obs} 个细胞")

# 对齐
idx = [list(adata.obs_names).index(c) for c in common]
gt_aligned = gt.loc[common, 'cell_type'].values
mapper_aligned = mapper_labels[idx]

# 给 Mapper 分量起可读名称
mapper_names = np.array([f'Comp_{l}' if l >= 0 else 'Unassigned'
                          for l in mapper_aligned])

# ============================================================
# 3. 混淆矩阵
# ============================================================
ct = pd.crosstab(
    pd.Series(gt_aligned, name='Cell Type'),
    pd.Series(mapper_names, name='Mapper Component'),
)

# 按 Mapper 分量编号排序列
comp_order = sorted([c for c in ct.columns if c.startswith('Comp_')],
                     key=lambda x: int(x.split('_')[1]))
if 'Unassigned' in ct.columns:
    comp_order.append('Unassigned')
ct = ct[comp_order]

# 按最大归属分量排序行（同一分量的类型靠在一起）
row_order = []
for comp in comp_order:
    types_in_comp = ct[comp].sort_values(ascending=False).index.tolist()
    for t in types_in_comp:
        if t not in row_order:
            row_order.append(t)
ct = ct.loc[row_order]

print("\n混淆矩阵 (Cell Type × Mapper Component):")
print(ct.to_string())

# ============================================================
# 4. 绘制混淆矩阵热力图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

# 行归一化（每行除以行总和，显示比例）
ct_norm = ct.div(ct.sum(axis=1), axis=0)

im = ax.imshow(ct_norm.values, cmap='YlOrRd', aspect='auto',
               vmin=0, vmax=1)

# 标注：左边数字是原始计数，右边百分比
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        count = ct.values[i, j]
        pct = ct_norm.values[i, j]
        if count > 0:
            color = 'white' if pct > 0.6 else 'black'
            ax.text(j, i, f'{count}\n({pct:.0%})', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

ax.set_xticks(range(ct.shape[1]))
ax.set_xticklabels(ct.columns, fontsize=11, fontweight='bold')
ax.set_yticks(range(ct.shape[0]))
ax.set_yticklabels(ct.index, fontsize=10)
ax.set_xlabel('Mapper Component', fontsize=13, fontweight='bold')
ax.set_ylabel('Ground Truth Cell Type', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix: Mapper Components vs Ground Truth\n'
             '(row-normalized, numbers = cell counts)',
             fontsize=14, fontweight='bold')

# 色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Proportion within cell type', fontsize=11)

# 分量分隔线
for j in range(1, ct.shape[1]):
    ax.axvline(j - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(figp('confusion_heatmap.png'), dpi=300, bbox_inches='tight')
print(f"\n[Info] Saved: {figp('confusion_heatmap.png')}")
plt.close()

# ============================================================
# 5. Benchmark 对比表
# ============================================================
true_k = len(set(gt_aligned))

# Mapper
ari_m = adjusted_rand_score(gt_aligned, mapper_aligned)
nmi_m = normalized_mutual_info_score(gt_aligned, mapper_aligned)

results = [{
    'Method': 'Mapper (TDA)',
    'k': len(set(mapper_names) - {'Unassigned'}),
    'Deviation': '',
    'ARI': round(ari_m, 4),
    'NMI': round(nmi_m, 4),
}]

# Leiden 多分辨率
if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, random_state=RANDOM_SEED)

for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}',
                 random_state=RANDOM_SEED)
    ll = adata.obs[f'leiden_{res}'].values[idx]
    results.append({
        'Method': f'Leiden (res={res})',
        'k': len(set(ll)),
        'Deviation': '',
        'ARI': round(adjusted_rand_score(gt_aligned, ll), 4),
        'NMI': round(normalized_mutual_info_score(gt_aligned, ll), 4),
    })

df = pd.DataFrame(results)
for i, r in df.iterrows():
    dev = (r['k'] - true_k) / true_k
    df.at[i, 'Deviation'] = f"{dev:+.0%}"

print(f"\n{'=' * 65}")
print(f"Benchmark vs Ground Truth ({true_k} cell types)")
print(f"{'=' * 65}")
print(df.to_string(index=False))

# Yu et al. 评级
print(f"\nYu et al. 评级:")
for _, r in df.iterrows():
    dev = abs(r['k'] - true_k) / true_k
    dg = 'Good' if dev <= 0.2 else ('Inter.' if dev <= 0.5 else 'Poor')
    ag = 'Good' if r['ARI'] >= 0.7 else ('Inter.' if r['ARI'] >= 0.5 else 'Poor')
    ng = 'Good' if r['NMI'] >= 0.7 else ('Inter.' if r['NMI'] >= 0.5 else 'Poor')
    print(f"  {r['Method']:20s}  k={r['k']:2d}  "
          f"Dev={r['Deviation']:5s}({dg:5s})  "
          f"ARI={r['ARI']:.4f}({ag:5s})  "
          f"NMI={r['NMI']:.4f}({ng:5s})")

# ============================================================
# 6. Benchmark 综合对比图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

methods = df['Method'].values
x = np.arange(len(methods))
colors = ['#e74c3c'] + ['#3498db'] * (len(methods) - 1)

# (A) 聚类数
ax = axes[0]
bars = ax.bar(x, df['k'].values, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(true_k, color='green', linestyle='--', linewidth=2,
           label=f'Ground Truth (k={true_k})')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Number of Clusters', fontsize=11)
ax.set_title('(A) Cluster Number Estimation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, df['k'].values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', fontsize=9, fontweight='bold')

# (B) ARI
ax = axes[1]
bars = ax.bar(x, df['ARI'].values, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.7)')
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Inter. threshold (0.5)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('ARI', fontsize=11)
ax.set_title('(B) Adjusted Rand Index', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

# (C) NMI
ax = axes[2]
bars = ax.bar(x, df['NMI'].values, color=colors, alpha=0.85, edgecolor='white')
ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold (0.7)')
ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Inter. threshold (0.5)')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('NMI', fontsize=11)
ax.set_title('(C) Normalized Mutual Information', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(figp('benchmark_summary.png'), dpi=300, bbox_inches='tight')
print(f"\n[Info] Saved: {figp('benchmark_summary.png')}")
plt.close()

# ============================================================
# 7. 每个分量的主导类型分析
# ============================================================
print(f"\n{'=' * 65}")
print("Mapper 分量 → 细胞类型映射")
print(f"{'=' * 65}")
for comp in comp_order:
    if comp == 'Unassigned':
        continue
    col = ct[comp]
    total = col.sum()
    dominant = col.idxmax()
    dom_pct = col.max() / total * 100
    types_present = col[col > 0].sort_values(ascending=False)
    print(f"\n  {comp} ({total} cells):")
    print(f"    主导类型: {dominant} ({dom_pct:.0f}%)")
    if len(types_present) > 1:
        print(f"    合并的类型:")
        for t, n in types_present.items():
            print(f"      {t}: {n} ({n/total*100:.0f}%)")

print(f"\n[Done]")
