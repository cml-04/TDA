"""
run_all.py — 一键运行全部实验
================================
放在项目根目录，调用 Core_code/ 中的脚本。
所有输出图片在 fig/ 各子目录中，日志保存在项目根目录。

目录结构:
  项目根/
  ├── run_all.py          ← 本文件
  ├── sce_dta.csv
  ├── cell_type_labels.csv
  ├── imbalance_data/
  ├── fig/                ← 输出图片
  │   ├── persistence/
  │   ├── benchmark/
  │   └── ...
  ├── cache/              ← 自动缓存
  └── Core_code/          ← 全部实验脚本
      ├── core.py
      ├── exp_persistence.py
      └── ...

用法:
  python run_all.py              # 运行全部
  python run_all.py --skip 4 5   # 跳过第4、5步
  python run_all.py --only 1 2   # 只运行第1、2步
"""

import sys
import os
import time
import subprocess
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Core_code')

# ============================================================
# 实验列表 (按论文章节排序)
# ============================================================
EXPERIMENTS = [
    # (编号, 脚本, 章节, 说明)
    (1,  'exp_persistence.py',   '4.1',     '持久同调 + 聚类数估计'),
    (2,  'benchmark.py',         '4.2',     '混淆矩阵 + ARI/NMI 评级'),
    (3,  'exp_topology.py',      '4.2/4.5', 'Mapper 拓扑图 + 稳定性'),
    (4,  'exp_ablation.py',      '4.3',     '插值消融 + Leiden 基准'),
    (5,  'magic_sensitivity.py', '3.4/4.3', 'MAGIC 参数 t 敏感性'),
    (6,  'imbalanced_full.py',   '4.4',     '12组不平衡实验'),
    (7,  'exp_rarity.py',        '4.5',     '稀有细胞评分 + 3方法对比'),
    (8,  'two_stage.py',         '4.2补充',  '两阶段联合聚类'),
    (9,  'exp_figures.py',       '全章',     '论文统一出图'),
    (10, 'group_analysis.py',    '4.2辅助',  'DE 热力图 + Excel'),
    (11, 'exp_supplement.py',    '全章补充',  '7张补充图 (流程图/UMAP对比/barcode/2×2框架)'),
    (12, 'exp_edge_expression.py','4.5核心',  '边缘节点表达谱比较 (热力图/折线/箱线/火山)'),
    (13, 'exp_gap_fixes.py',     '全章修补',  '6个缺口修补 (两阶段/不平衡深入/GT着色/髓系/总表/阈值)'),
]


def run_experiment(num, script, section, desc, log_file, python_exe):
    """运行单个实验，输出同时写入终端和日志"""
    header = (
        f"\n{'█' * 70}\n"
        f"█  [{num:2d}/13] {script:25s}  章节 {section:8s}\n"
        f"█  {desc}\n"
        f"{'█' * 70}\n"
    )
    print(header)
    log_file.write(header)
    log_file.flush()

    if not os.path.exists(os.path.join(SCRIPT_DIR, script)):
        msg = f"  ⚠ 文件不存在: Core_code/{script}, 跳过\n"
        print(msg)
        log_file.write(msg)
        log_file.flush()
        return False, 0

    t0 = time.time()
    script_path = os.path.join(SCRIPT_DIR, script)
    try:
        proc = subprocess.Popen(
            [python_exe, '-u', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        proc.wait()
        elapsed = time.time() - t0
        rc = proc.returncode

        if rc == 0:
            status = f"\n  ✓ 完成 ({elapsed:.1f}s)\n"
        else:
            status = f"\n  ✗ 失败 (exit code {rc}, {elapsed:.1f}s)\n"
        print(status)
        log_file.write(status)
        log_file.flush()
        return rc == 0, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        err = f"\n  ✗ 异常: {e} ({elapsed:.1f}s)\n"
        print(err)
        log_file.write(err)
        log_file.flush()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description='一键运行全部实验')
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                        help='跳过指定编号的实验 (如 --skip 6 10)')
    parser.add_argument('--only', type=int, nargs='+', default=[],
                        help='只运行指定编号的实验 (如 --only 1 2 3)')
    parser.add_argument('--log', type=str, default=None,
                        help='日志文件名 (默认: output_YYYYMMDD_HHMMSS.log)')
    args = parser.parse_args()

    # 日志文件名
    if args.log:
        log_name = args.log
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f'output_{ts}.log'

    # 确定运行哪些实验
    if args.only:
        to_run = [e for e in EXPERIMENTS if e[0] in args.only]
    else:
        to_run = [e for e in EXPERIMENTS if e[0] not in args.skip]

    # Python 解释器
    python_exe = sys.executable

    # 开始
    os.makedirs('fig', exist_ok=True)

    with open(log_name, 'w', encoding='utf-8') as log_file:
        start_msg = (
            f"{'=' * 70}\n"
            f"  TDA 单细胞聚类论文 — 全部实验运行\n"
            f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Python: {python_exe}\n"
            f"  计划运行: {len(to_run)}/13 个实验\n"
            f"  日志文件: {log_name}\n"
            f"{'=' * 70}\n\n"
            f"实验列表:\n"
        )
        for num, script, section, desc in to_run:
            start_msg += f"  [{num:2d}] {script:25s} → {section:8s} {desc}\n"
        start_msg += "\n"

        print(start_msg)
        log_file.write(start_msg)
        log_file.flush()

        # 逐个运行
        results = []
        total_t0 = time.time()

        for num, script, section, desc in to_run:
            success, elapsed = run_experiment(
                num, script, section, desc, log_file, python_exe)
            results.append((num, script, section, desc, success, elapsed))

        total_elapsed = time.time() - total_t0

        # 汇总
        summary = (
            f"\n{'=' * 70}\n"
            f"  运行汇总\n"
            f"{'=' * 70}\n\n"
        )
        n_ok = sum(1 for r in results if r[4])
        n_fail = sum(1 for r in results if not r[4])

        for num, script, section, desc, success, elapsed in results:
            icon = "✓" if success else "✗"
            summary += (f"  {icon} [{num:2d}] {script:25s} "
                        f"{section:8s} {elapsed:6.1f}s  {desc}\n")

        summary += (
            f"\n  总计: {n_ok} 成功, {n_fail} 失败, "
            f"总耗时 {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)\n"
            f"  日志: {log_name}\n"
            f"  图片: fig/ 目录\n"
        )

        # 列出生成的图片 (含子目录)
        if os.path.exists('fig'):
            all_pngs = []
            for root, dirs, files in os.walk('fig'):
                for f in sorted(files):
                    if f.endswith('.png') or f.endswith('.html'):
                        all_pngs.append(os.path.join(root, f))
            summary += f"\n  生成了 {len(all_pngs)} 个文件:\n"
            current_dir = None
            for p in sorted(all_pngs):
                d = os.path.dirname(p)
                if d != current_dir:
                    current_dir = d
                    summary += f"\n    [{d}/]\n"
                summary += f"      {os.path.basename(p)}\n"

        summary += f"\n{'=' * 70}\n"

        print(summary)
        log_file.write(summary)

    print(f"\n全部输出已保存到: {log_name}")


if __name__ == '__main__':
    main()
