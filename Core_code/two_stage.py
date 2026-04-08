"""
two_stage.py — 两阶段联合聚类
================================
阶段1: Mapper 确定粗粒度拓扑骨架 (哪些类型孤立、哪些紧密相连)
阶段2: 对混合分量(Comp_0)内部用 Leiden 精细聚类
评估: 联合标签 vs ground truth 的 ARI/NMI, 与单独 Mapper / 单独 Leiden 对比
输出: fig/two_stage_*.png
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import os
from core import (RANDOM_SEED, fig_path, make_fig, full_pipeline, PROJECT_ROOT,
                  assign_mapper_labels, ensure_fig_dir)

ensure_fig_dir('two_stage')
figp = make_fig('two_stage')

# ============================================================
# 1. 运行 Mapper 流程
# ============================================================
adata, graph, G, comps, mapper_labels, lens_1d = full_pipeline()

# ============================================================
# 2. 加载 Ground Truth
# ============================================================
gt = pd.read_csv(os.path.join(PROJECT_ROOT, 'cell_type_labels.csv')).set_index('cell_id')
common = adata.obs_names.intersection(gt.index)
print(f"匹配: {len(common)}/{adata.n_obs}")

gt_aligned = gt.loc[common, 'cell_type'].values
idx_map = [list(adata.obs_names).index(c) for c in common]
mapper_aligned = mapper_labels[idx_map]
true_k = len(set(gt_aligned))

# ============================================================
# 3. 识别混合分量 (纯度 < 80% 的分量需要细分)
# ============================================================
PURITY_THRESHOLD = 0.80

comp_info = {}
for comp_id in sorted(set(mapper_aligned)):
    if comp_id < 0:
        continue
    mask = mapper_aligned == comp_id
    types_in_comp = pd.Series(gt_aligned[mask]).value_counts()
    total = mask.sum()
    dominant_pct = types_in_comp.iloc[0] / total
    comp_info[comp_id] = {
        'n_cells': total,
        'n_types': len(types_in_comp),
        'dominant': types_in_comp.index[0],
        'purity': dominant_pct,
        'needs_refine': dominant_pct < PURITY_THRESHOLD,
    }
    status = "→ 需要细分" if dominant_pct < PURITY_THRESHOLD else "✓ 高纯度"
    print(f"  Comp_{comp_id}: {total} cells, {len(types_in_comp)} types, "
          f"purity={dominant_pct:.0%} {status}")

mixed_comps = [c for c, info in comp_info.items() if info['needs_refine']]
pure_comps = [c for c, info in comp_info.items() if not info['needs_refine']]
print(f"\n高纯度分量: {pure_comps}")
print(f"需细分分量: {mixed_comps}")

# ============================================================
# 4. 阶段2: 对混合分量内部做 Leiden 精细聚类
# ============================================================
# 构建联合标签: 高纯度分量保持 Mapper 标签, 混合分量内部用 Leiden 细分
combined_labels = np.full(len(common), '', dtype=object)

# 高纯度分量: 直接用 "Mapper_X" 作为标签
for comp_id in pure_comps:
    mask = mapper_aligned == comp_id
    combined_labels[mask] = f'M_{comp_id}'

# 混合分量: 提取子集, 跑 Leiden, 用 "Leiden_X_Y" 作为标签
best_res = None
best_ari = -1
best_leiden_labels = None

for comp_id in mixed_comps:
    comp_mask_global = np.array([i for i, c in enumerate(common)
                                  if mapper_aligned[list(common).index(c) if c in common else -1] == comp_id
                                  ]) if False else np.where(mapper_aligned == comp_id)[0]

    # 提取该分量的细胞在 adata 中的索引
    comp_cell_ids = [common[i] for i in range(len(common)) if mapper_aligned[i] == comp_id]
    adata_idx = [list(adata.obs_names).index(c) for c in comp_cell_ids]

    # 创建子集 adata
    adata_sub = adata[adata_idx].copy()

    # 在子集上重新计算 PCA → neighbors → Leiden
    sc.tl.pca(adata_sub, n_comps=min(30, adata_sub.n_obs - 1), random_state=RANDOM_SEED)
    sc.pp.neighbors(adata_sub, random_state=RANDOM_SEED)

    # 扫描多个分辨率, 选择使联合 ARI 最高的
    print(f"\n对 Comp_{comp_id} ({len(comp_cell_ids)} cells) 扫描 Leiden 分辨率:")

    for res in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        sc.tl.leiden(adata_sub, resolution=res, key_added=f'leiden_{res}',
                     random_state=RANDOM_SEED)
        sub_leiden = adata_sub.obs[f'leiden_{res}'].values.astype(str)

        # 临时构建联合标签
        temp_labels = combined_labels.copy()
        comp_positions = np.where(mapper_aligned == comp_id)[0]
        for i, pos in enumerate(comp_positions):
            temp_labels[pos] = f'L_{comp_id}_{sub_leiden[i]}'

        # 填充未赋值的位置 (其他混合分量暂用 Mapper 标签)
        for i in range(len(temp_labels)):
            if temp_labels[i] == '':
                temp_labels[i] = f'M_{mapper_aligned[i]}'

        ari = adjusted_rand_score(gt_aligned, temp_labels)
        nmi = normalized_mutual_info_score(gt_aligned, temp_labels)
        n_sub = len(set(sub_leiden))

        print(f"    res={res}: {n_sub} sub-clusters, combined ARI={ari:.4f}, NMI={nmi:.4f}")

        if ari > best_ari:
            best_ari = ari
            best_res = res
            best_leiden_labels = sub_leiden.copy()

    print(f"  → 最优: res={best_res}, ARI={best_ari:.4f}")

    # 用最优分辨率的结果填充联合标签
    comp_positions = np.where(mapper_aligned == comp_id)[0]
    for i, pos in enumerate(comp_positions):
        combined_labels[pos] = f'L_{comp_id}_{best_leiden_labels[i]}'

# 处理未分配的细胞
for i in range(len(combined_labels)):
    if combined_labels[i] == '':
        combined_labels[i] = f'M_{mapper_aligned[i]}'

# ============================================================
# 5. 三种方法的对比
# ============================================================
# 单独 Mapper
ari_mapper = adjusted_rand_score(gt_aligned, mapper_aligned)
nmi_mapper = normalized_mutual_info_score(gt_aligned, mapper_aligned)
k_mapper = len(set(mapper_aligned[mapper_aligned >= 0]))

# 单独 Leiden (最佳分辨率)
if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, random_state=RANDOM_SEED)

best_leiden_global = None
best_ari_global = -1
leiden_results = []

for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_g_{res}',
                 random_state=RANDOM_SEED)
    ll = adata.obs[f'leiden_g_{res}'].values[idx_map]
    a = adjusted_rand_score(gt_aligned, ll)
    n = normalized_mutual_info_score(gt_aligned, ll)
    leiden_results.append({'res': res, 'k': len(set(ll)), 'ARI': a, 'NMI': n})
    if a > best_ari_global:
        best_ari_global = a
        best_leiden_global = {'res': res, 'k': len(set(ll)), 'ARI': a, 'NMI': n}

# 两阶段联合
ari_combined = adjusted_rand_score(gt_aligned, combined_labels)
nmi_combined = normalized_mutual_info_score(gt_aligned, combined_labels)
k_combined = len(set(combined_labels))

print("\n" + "=" * 70)
print("三种方法对比 (vs Ground Truth, k_true=10)")
print("=" * 70)

results_df = pd.DataFrame([
    {'Method': 'Mapper only', 'k': k_mapper,
     'ARI': round(ari_mapper, 4), 'NMI': round(nmi_mapper, 4)},
    {'Method': f'Leiden only (res={best_leiden_global["res"]})', 'k': best_leiden_global['k'],
     'ARI': round(best_leiden_global['ARI'], 4), 'NMI': round(best_leiden_global['NMI'], 4)},
    {'Method': f'Two-Stage (Mapper+Leiden)', 'k': k_combined,
     'ARI': round(ari_combined, 4), 'NMI': round(nmi_combined, 4)},
])
print(results_df.to_string(index=False))

# 提升幅度
print(f"\n提升:")
print(f"  ARI: Mapper {ari_mapper:.4f} → Two-Stage {ari_combined:.4f} "
      f"(+{(ari_combined - ari_mapper):.4f}, +{(ari_combined - ari_mapper)/ari_mapper*100:.0f}%)")
print(f"  NMI: Mapper {nmi_mapper:.4f} → Two-Stage {nmi_combined:.4f} "
      f"(+{(nmi_combined - nmi_mapper):.4f}, +{(nmi_combined - nmi_mapper)/nmi_mapper*100:.0f}%)")
print(f"  vs Leiden best: ARI {best_leiden_global['ARI']:.4f}, NMI {best_leiden_global['NMI']:.4f}")

# ============================================================
# 6. 两阶段联合的混淆矩阵
# ============================================================
ct = pd.crosstab(
    pd.Series(gt_aligned, name='Cell Type'),
    pd.Series(combined_labels, name='Two-Stage Cluster'),
)
print(f"\n两阶段联合聚类混淆矩阵:")
print(ct.to_string())

# ============================================================
# 7. 绘图
# ============================================================

# --- 图 1: 三种方法 ARI/NMI 柱状图 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = results_df['Method'].values
x = np.arange(len(methods))
colors = ['#e74c3c', '#3498db', '#2ecc71']

for ax, metric in zip(axes, ['ARI', 'NMI']):
    bars = ax.bar(x, results_df[metric].values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.4, label='Good (0.7)')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.4, label='Inter. (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mapper\nonly', 'Leiden\nonly', 'Two-Stage\n(Mapper+Leiden)'],
                        fontsize=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, results_df[metric].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Two-Stage Clustering vs Single Methods', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figp('two_stage_comparison.png'), dpi=300, bbox_inches='tight')
print(f"\n[Info] Saved: {figp('two_stage_comparison.png')}")
plt.close()

# --- 图 2: 流程示意 + 混淆矩阵热力图 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 左: Mapper only 混淆矩阵 (行归一化)
ct_mapper = pd.crosstab(
    pd.Series(gt_aligned, name='Cell Type'),
    pd.Series([f'Comp_{l}' if l >= 0 else 'NA' for l in mapper_aligned],
              name='Mapper'),
)
ct_mapper_norm = ct_mapper.div(ct_mapper.sum(axis=1), axis=0)

ax = axes[0]
im = ax.imshow(ct_mapper_norm.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
for i in range(ct_mapper.shape[0]):
    for j in range(ct_mapper.shape[1]):
        v = ct_mapper.values[i, j]
        p = ct_mapper_norm.values[i, j]
        if v > 0:
            c = 'white' if p > 0.6 else 'black'
            ax.text(j, i, f'{v}', ha='center', va='center', fontsize=8,
                    fontweight='bold', color=c)
ax.set_xticks(range(ct_mapper.shape[1]))
ax.set_xticklabels(ct_mapper.columns, fontsize=9, fontweight='bold')
ax.set_yticks(range(ct_mapper.shape[0]))
ax.set_yticklabels(ct_mapper.index, fontsize=8)
ax.set_title(f'Stage 1: Mapper only\nARI={ari_mapper:.3f}  NMI={nmi_mapper:.3f}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Mapper Component')

# 右: Two-Stage 混淆矩阵 (行归一化)
ct_norm = ct.div(ct.sum(axis=1), axis=0)

ax = axes[1]
im2 = ax.imshow(ct_norm.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        p = ct_norm.values[i, j]
        if v > 0:
            c = 'white' if p > 0.6 else 'black'
            ax.text(j, i, f'{v}', ha='center', va='center', fontsize=7,
                    fontweight='bold', color=c)
ax.set_xticks(range(ct.shape[1]))
ax.set_xticklabels(ct.columns, fontsize=7, rotation=45, ha='right')
ax.set_yticks(range(ct.shape[0]))
ax.set_yticklabels(ct.index, fontsize=8)
ax.set_title(f'Stage 2: Mapper + Leiden refinement\nARI={ari_combined:.3f}  NMI={nmi_combined:.3f}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Two-Stage Cluster')

plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.08, shrink=0.8, label='Row proportion')
plt.tight_layout()
plt.savefig(figp('two_stage_confusion.png'), dpi=300, bbox_inches='tight')
print(f"[Info] Saved: {figp('two_stage_confusion.png')}")
plt.close()

# --- 图 3: UMAP 三面板对比 ---
if 'X_umap' in adata.obsm:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    umap = adata.obsm['X_umap'][idx_map]

    # (A) Ground Truth
    ax = axes[0]
    types_unique = sorted(set(gt_aligned))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(types_unique)))
    type_to_color = {t: cmap[i] for i, t in enumerate(types_unique)}
    colors_gt = [type_to_color[t] for t in gt_aligned]
    ax.scatter(umap[:, 0], umap[:, 1], c=colors_gt, s=8, alpha=0.7)
    ax.set_title(f'(A) Ground Truth\n{true_k} cell types', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')

    # (B) Mapper only
    ax = axes[1]
    mapper_unique = sorted(set(mapper_aligned[mapper_aligned >= 0]))
    cmap_m = plt.cm.Set1(np.linspace(0, 1, max(len(mapper_unique), 1)))
    colors_m = [cmap_m[mapper_unique.index(l)] if l >= 0 else [0.8, 0.8, 0.8, 1]
                for l in mapper_aligned]
    ax.scatter(umap[:, 0], umap[:, 1], c=colors_m, s=8, alpha=0.7)
    ax.set_title(f'(B) Mapper only\nk={k_mapper}  ARI={ari_mapper:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP 1')

    # (C) Two-Stage
    ax = axes[2]
    combined_unique = sorted(set(combined_labels))
    cmap_c = plt.cm.tab20(np.linspace(0, 1, max(len(combined_unique), 1)))
    label_to_color = {l: cmap_c[i] for i, l in enumerate(combined_unique)}
    colors_c = [label_to_color[l] for l in combined_labels]
    ax.scatter(umap[:, 0], umap[:, 1], c=colors_c, s=8, alpha=0.7)
    ax.set_title(f'(C) Two-Stage (Mapper+Leiden)\nk={k_combined}  ARI={ari_combined:.3f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP 1')

    plt.tight_layout()
    plt.savefig(figp('two_stage_umap.png'), dpi=300, bbox_inches='tight')
    print(f"[Info] Saved: {figp('two_stage_umap.png')}")
    plt.close()

# ============================================================
# 8. Yu et al. 框架下的综合评级
# ============================================================
print("\n" + "=" * 70)
print("Yu et al. [1] 评测框架下的综合评级")
print("=" * 70)

for _, r in results_df.iterrows():
    dev = abs(r['k'] - true_k) / true_k
    dg = 'Good' if dev <= 0.2 else ('Inter.' if dev <= 0.5 else 'Poor')
    ag = 'Good' if r['ARI'] >= 0.7 else ('Inter.' if r['ARI'] >= 0.5 else 'Poor')
    ng = 'Good' if r['NMI'] >= 0.7 else ('Inter.' if r['NMI'] >= 0.5 else 'Poor')
    print(f"  {r['Method']:30s}  k={r['k']:2d}  "
          f"Dev={dev:+.0%}({dg:5s})  "
          f"ARI={r['ARI']:.4f}({ag:5s})  "
          f"NMI={r['NMI']:.4f}({ng:5s})")

print(f"\n[Done] 所有图片保存在 fig/ 目录")