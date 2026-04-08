"""
core.py — 共享基础模块 (带缓存)
================================
缓存策略:
  cache/normalized.h5ad  — load_and_normalize() 的结果
  cache/pipeline.h5ad    — full_pipeline() 中 MAGIC+PCA+UMAP 后的 adata
  Mapper 不缓存 (< 1 秒, 且参数可能变)

缓存自动失效条件:
  - sce_dta.csv 被修改 (mtime 比缓存新)
  - 手动调用 clear_cache()
  - 删除 cache/ 目录
"""

import os
import time
import json
import numpy as np
import pandas as pd
import scanpy as sc
import magic
import networkx as nx
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
from sklearn.cluster import DBSCAN
from gtda.homology import VietorisRipsPersistence
import kmapper as km
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局常量
# ============================================================
RANDOM_SEED = 42

# 项目根目录 = Core_code 的上一级
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, 'sce_dta.csv')
FIG_DIR = os.path.join(PROJECT_ROOT, 'fig')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

np.random.seed(RANDOM_SEED)


# ============================================================
# 目录管理
# ============================================================
def ensure_fig_dir(subdir=None):
    d = os.path.join(FIG_DIR, subdir) if subdir else FIG_DIR
    os.makedirs(d, exist_ok=True)

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def fig_path(name, subdir=None):
    """返回 fig/name 或 fig/subdir/name, 自动创建目录"""
    if subdir:
        d = os.path.join(FIG_DIR, subdir)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name)
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def make_fig(subdir):
    """返回绑定了子目录的 fig_path 函数, 供各脚本使用:
       figp = make_fig('persistence')
       figp('barcode.png')  →  'fig/persistence/barcode.png'
    """
    def _fig_path(name):
        return fig_path(name, subdir=subdir)
    return _fig_path

def cache_path(name):
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, name)


# ============================================================
# 缓存工具
# ============================================================
def _source_mtime(path=DATA_PATH):
    """获取源数据文件的修改时间"""
    if os.path.exists(path):
        return os.path.getmtime(path)
    return 0

def _cache_valid(cache_file, source_path=DATA_PATH):
    """检查缓存是否有效: 存在 且 比源文件新"""
    if not os.path.exists(cache_file):
        return False
    return os.path.getmtime(cache_file) > _source_mtime(source_path)

def clear_cache():
    """清除所有缓存"""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("[Cache] Cleared.")
    else:
        print("[Cache] Nothing to clear.")


# ============================================================
# 数据加载与 QC (带缓存)
# ============================================================
def load_and_normalize(path=DATA_PATH, use_cache=True):
    """
    加载 CSV → QC → normalize → log1p
    缓存为 cache/normalized.h5ad, 后续调用直接读取 (~0.3s vs ~3s)
    """
    cf = cache_path('normalized.h5ad')

    if use_cache and _cache_valid(cf, path):
        print(f"[Cache] Loading normalized data from {cf}")
        adata = sc.read_h5ad(cf)
        print(f"[Core] Cached normalized: {adata.shape}")
        return adata

    print(f"[Core] Loading from CSV: {path}")
    adata = sc.read_csv(path).T
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = (adata.var_names.str.startswith('MT-') |
                       adata.var_names.str.startswith('mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],
                                percent_top=None, log1p=False, inplace=True)
    n_mt = adata.var['mt'].sum()
    if n_mt > 0:
        adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
    else:
        adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if use_cache:
        adata.write_h5ad(cf)
        print(f"[Cache] Saved normalized → {cf}")

    print(f"[Core] Loaded & normalized: {adata.shape}")
    return adata


# ============================================================
# MAGIC 插值
# ============================================================
def apply_magic(adata_log, t=3, n_hvg=2000):
    """HVG → sqrt → MAGIC, 返回 adata_magic (raw = log-normalized 全基因)"""
    raw_adata = adata_log.copy()

    sc.pp.highly_variable_genes(adata_log, n_top_genes=n_hvg,
                                 flavor='seurat', subset=False)
    hvg_mask = adata_log.var.highly_variable
    print(f"[Core] {hvg_mask.sum()} HVGs selected.")

    work = adata_log.copy()
    if issparse(work.X):
        work.X = np.expm1(work.X.toarray())
    else:
        work.X = np.expm1(work.X)
    work.X = np.sqrt(work.X)

    print(f"[Core] Running MAGIC (t={t})...")
    magic_op = magic.MAGIC(t=t, n_pca=20, n_jobs=-1, random_state=RANDOM_SEED)
    work.X = magic_op.fit_transform(work.to_df()).values

    out = work[:, hvg_mask].copy()
    out.raw = raw_adata
    print(f"[Core] Imputed shape: {out.shape}")
    return out


# ============================================================
# Auto DBSCAN eps
# ============================================================
def auto_eps(data, k=15, percentile=90):
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(data)
    d, _ = nn.kneighbors(data)
    eps = np.percentile(d[:, -1], percentile)
    print(f"[Core] Auto eps: k={k}, p{percentile} = {eps:.3f}")
    return eps


# ============================================================
# Mapper 构建 (不缓存, < 1 秒)
# ============================================================
def build_mapper(pca_data, lens_1d):
    """1D lens 自动调参 Mapper → (graph, G, comps, eps)"""
    if lens_1d.ndim == 1:
        lens_1d = lens_1d.reshape(-1, 1)

    eps = auto_eps(pca_data)
    configs = [
        {'n_cubes': 10, 'perc_overlap': 0.5},
        {'n_cubes': 8,  'perc_overlap': 0.6},
        {'n_cubes': 6,  'perc_overlap': 0.7},
    ]
    for cp in configs:
        kmap = km.KeplerMapper(verbose=0)
        graph = kmap.map(lens_1d, X=pca_data, cover=km.Cover(**cp),
                          clusterer=DBSCAN(eps=eps, min_samples=3))
        G = nx.Graph()
        for nid, mem in graph['nodes'].items():
            G.add_node(nid, members=mem)
        for src, tgts in graph['links'].items():
            for t in tgts:
                G.add_edge(src, t)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        n_nodes = G.number_of_nodes()
        largest = len(comps[0]) if comps else 0
        print(f"  [Mapper] cover={cp}, eps={eps:.1f} → "
              f"{n_nodes} nodes, {len(comps)} comps, largest={largest}")
        if largest >= n_nodes * 0.4:
            return graph, G, comps, eps
        eps *= 1.3
    return graph, G, comps, eps


def assign_mapper_labels(graph, comps, n_cells):
    """按连通分量分配硬标签, -1 = 未覆盖"""
    labels = np.full(n_cells, -1, dtype=int)
    for ci, comp in enumerate(comps):
        for nid in comp:
            for idx in graph['nodes'][nid]:
                if idx < n_cells:
                    labels[idx] = ci
    return labels


# ============================================================
# 全流程 (带缓存)
# ============================================================
def full_pipeline(path=DATA_PATH, t=3, use_cache=True):
    """
    完整流程: load → MAGIC → PCA → UMAP → Mapper
    
    缓存策略:
      - MAGIC + PCA + UMAP 的结果缓存为 cache/pipeline.h5ad (~20s → ~1s)
      - Mapper 每次重建 (< 1s, 参数可能调整)
    
    返回: (adata, graph, G, comps, mapper_labels, lens_1d)
    """
    cf = cache_path('pipeline.h5ad')
    t0 = time.time()

    if use_cache and _cache_valid(cf, path):
        print(f"[Cache] Loading pipeline data from {cf}")
        adata = sc.read_h5ad(cf)
        print(f"[Cache] Loaded in {time.time() - t0:.1f}s: {adata.shape}")
    else:
        # 完整计算
        adata_log = load_and_normalize(path, use_cache=use_cache)
        adata = apply_magic(adata_log, t=t)

        # 同步 QC 指标
        for col in ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']:
            if col in adata_log.obs:
                adata.obs[col] = adata_log.obs[col].values

        sc.tl.pca(adata, n_comps=30, random_state=RANDOM_SEED)
        sc.pp.neighbors(adata, random_state=RANDOM_SEED)
        sc.tl.umap(adata, random_state=RANDOM_SEED)

        # 保存缓存
        if use_cache:
            adata.write_h5ad(cf)
            print(f"[Cache] Saved pipeline → {cf}")
            print(f"[Cache] Next run will load in < 1s instead of ~{time.time()-t0:.0f}s")

    # Mapper 每次重建 (快速, 且参数可能变)
    lens_1d = adata.obsm['X_umap'][:, 0]
    graph, G, comps, eps = build_mapper(adata.obsm['X_pca'], lens_1d)
    labels = assign_mapper_labels(graph, comps, adata.n_obs)

    total = time.time() - t0
    print(f"\n[Core] Pipeline done in {total:.1f}s: {len(comps)} components, "
          f"{len(graph['nodes'])} nodes, {(labels >= 0).sum()}/{adata.n_obs} cells")

    return adata, graph, G, comps, labels, lens_1d
