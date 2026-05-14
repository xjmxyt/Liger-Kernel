#!/usr/bin/env python3
"""
Parse a benchmark CSV and print a Liger vs cuTile (tilegym) comparison summary.

Usage:
    python parse_benchmark_csv.py benchmark/data/cutile/all_benchmark_data_tilegym.csv
    python parse_benchmark_csv.py benchmark/data/all_benchmark_data_tilegym.csv --kernels layer_norm,geglu
    python parse_benchmark_csv.py benchmark/data/all_benchmark_data_tilegym.csv --verbose
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_GREEN = "\033[92m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

METRIC_ORDER = ("speed", "memory")
MODE_ORDER = ("forward", "backward", "full", "no-grad-forward")


def parse_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def color_ratio(ratio, fmt=".2f", tol=0.01):
    if ratio is None:
        return "-"
    s = f"{ratio:{fmt}}x"
    if abs(ratio - 1.0) <= tol:
        return s  # within 1% — no color
    color = _GREEN if ratio > 1.0 else _RED
    return f"{color}{s}{_RESET}"


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows, providers=("liger", "tilegym")):
    """Group by (gpu, kernel, mode, metric, x_value, extra_config) → {provider: row}."""
    grouped = defaultdict(dict)
    for row in rows:
        if row["kernel_provider"] not in providers:
            continue
        key = (
            row["gpu_name"],
            row["kernel_name"],
            row["kernel_operation_mode"],
            row["metric_name"],
            row["metric_unit"],
            row["x_name"],
            row["x_value"],
            row.get("extra_benchmark_config_str", ""),
        )
        grouped[key][row["kernel_provider"]] = row
    return grouped


def geomean(vals):
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ──────────────────────────────────────────────────────────────
# Verbose: full per-x table
# ──────────────────────────────────────────────────────────────

def print_verbose_table(kernel, metric, unit, x_name, mode_data):
    modes = [m for m in MODE_ORDER if any(m in by_mode for by_mode in mode_data.values())]
    if not modes:
        return

    label = "LATENCY" if metric == "speed" else "MEMORY"
    print(f"\n  {_BOLD}{_CYAN}── {kernel} {label} ({unit}) ──────────────────────────────────────{_RESET}\n")

    x_vals = sorted(mode_data.keys(), key=lambda v: (int(v) if v.isdigit() else v))
    x_w = max(len(x_name), len("geomean"), max(len(v) for v in x_vals))
    val_w = 11
    ratio_w = 19

    hdr = [f"{x_name:>{x_w}}"]
    sep = ["─" * x_w]
    for m in modes:
        ms = m[:3]
        hdr += [f"{'liger ' + ms:>{val_w}}", f"{'cutile ' + ms:>{val_w}}", f"{'cuTile ' + ms + ' speedup':>{ratio_w}}"]
        sep += ["─" * val_w, "─" * val_w, "─" * ratio_w]
    print("  " + "  ".join(hdr))
    print("  " + "  ".join(sep))

    ratios_by_mode = defaultdict(list)
    for xv in x_vals:
        row_cells = [f"{xv:>{x_w}}"]
        for m in modes:
            pair = mode_data[xv].get(m, {})
            lrow = pair.get("liger")
            trow = pair.get("tilegym")
            ly = parse_float(lrow["y_value_50"]) if lrow else None
            ty = parse_float(trow["y_value_50"]) if trow else None
            ratio = (ly / ty) if (ly and ty and ty > 0) else None
            if ratio:
                ratios_by_mode[m].append(ratio)
            liger_best = ly is not None and (ty is None or ly <= ty)
            cutile_best = ty is not None and (ly is None or ty <= ly)

            def fmt(v, best):
                if v is None:
                    return f"{'-':>{val_w}}"
                s = f"{v:.3f}"
                return f"{_GREEN}{_BOLD}{s:>{val_w}}{_RESET}" if best else f"{s:>{val_w}}"

            row_cells += [fmt(ly, liger_best), fmt(ty, cutile_best), f"{color_ratio(ratio):>{ratio_w}}"]
        print("  " + "  ".join(row_cells))

    print("  " + "  ".join(sep))
    gm_cells = [f"{_BOLD}{'geomean':>{x_w}}{_RESET}"]
    for m in modes:
        gm = geomean(ratios_by_mode[m])
        gm_cells += [f"{_BOLD}{'-':>{val_w}}{_RESET}", f"{_BOLD}{'-':>{val_w}}{_RESET}",
                     f"{_BOLD}{color_ratio(gm):>{ratio_w}}{_RESET}"]
    print("  " + "  ".join(gm_cells))


# ──────────────────────────────────────────────────────────────
# Summary: one row per kernel — min/median/max speedup + avg memory ratio
# ──────────────────────────────────────────────────────────────

def median(vals):
    s = sorted(v for v in vals if v is not None)
    if not s:
        return None
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def print_summary_table(kernel_stats, gpu_name=None):
    """
    kernel_stats: {kernel: {metric: {mode: [ratios]}}}
    Shows forward and backward speed separately, plus mem ratio.
    """
    kernels = sorted(kernel_stats)
    if not kernels:
        return

    k_w = max(len("kernel"), max(len(k) for k in kernels))
    col_w = 8

    # detect which modes are present across all kernels
    all_modes = []
    for m in ("forward", "backward"):
        if any(m in kernel_stats[k].get("speed", {}) for k in kernels):
            all_modes.append(m)

    def cr(v):
        if v is None:
            return f"{'N/A':>{col_w}}"
        raw = color_ratio(v)
        plain = f"{v:.2f}x"
        pad = col_w - len(plain)
        return " " * max(pad, 0) + raw

    # build header columns: for each mode: min/med/max; then mem ratio
    mode_labels = {"forward": "fwd", "backward": "bwd"}
    subhdrs = []
    for mode in all_modes:
        ms = mode_labels[mode]
        subhdrs += [f"{ms+' min':>{col_w}}", f"{ms+' med':>{col_w}}", f"{ms+' max':>{col_w}}"]
    subhdrs += [f"{'mem fwd':>{col_w}}", f"{'mem bwd':>{col_w}}", f"{'mem full':>{col_w}}"]

    header = f"  {'kernel':<{k_w}}  " + "  ".join(subhdrs)
    total_w = k_w + len(all_modes) * 3 * (col_w + 2) + 3 * (col_w + 2) + 2
    sep = "  " + "─" * total_w

    title = "Liger vs cuTile — Summary"
    if gpu_name:
        title += f" [{gpu_name}]"
    print(f"\n{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"  ratio = liger / cuTile  │  speed: >1x = cuTile faster  │  mem ratio: >1x = cuTile uses less memory")
    print(sep)
    print(header)
    print(sep)

    for kernel in kernels:
        cells = [f"{kernel:<{k_w}}"]
        for mode in all_modes:
            ratios = kernel_stats[kernel].get("speed", {}).get(mode, [])
            cells += [
                cr(min(ratios) if ratios else None),
                cr(median(ratios)),
                cr(max(ratios) if ratios else None),
            ]
        # memory: show fwd / bwd / full independently
        mem = kernel_stats[kernel].get("memory", {})
        for mmode in ("forward", "backward", "full"):
            ratios = mem.get(mmode, [])
            avg = (sum(ratios) / len(ratios)) if ratios else None
            cells.append(cr(avg) if avg is not None else f"{'':>{col_w}}")
        print("  " + "  ".join(cells))

    print(sep)


# ──────────────────────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────────────────────

def build_kernel_stats(rows, filter_kernels=None, filter_gpus=None):
    """Collect speedup ratios per kernel, metric, and mode."""
    grouped = group_rows(rows)
    # stats[gpu][kernel][metric][mode] = [ratios]
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for key, providers in grouped.items():
        gpu, kernel, mode, metric, unit, x_name, x_value, _ = key
        if filter_gpus and gpu not in filter_gpus:
            continue
        if filter_kernels and kernel not in filter_kernels:
            continue
        lrow = providers.get("liger")
        trow = providers.get("tilegym")
        ly = parse_float(lrow["y_value_50"]) if lrow else None
        ty = parse_float(trow["y_value_50"]) if trow else None
        if ly and ty and ty > 0:
            stats[gpu][kernel][metric][mode].append(ly / ty)

    return stats


def build_verbose_tables(rows, filter_kernels=None, filter_metric=None, filter_gpus=None):
    grouped = group_rows(rows)
    tables = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    meta = {}

    for key, providers in grouped.items():
        gpu, kernel, mode, metric, unit, x_name, x_value, _ = key
        if filter_gpus and gpu not in filter_gpus:
            continue
        if filter_kernels and kernel not in filter_kernels:
            continue
        if filter_metric and metric != filter_metric:
            continue
        tables[gpu][kernel][metric][x_value][mode] = providers
        meta[(gpu, kernel, metric)] = (x_name, unit)

    return tables, meta


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark CSV and show Liger vs cuTile summary.")
    parser.add_argument("csv", help="Path to benchmark CSV file")
    parser.add_argument("--kernels", default=None, help="Comma-separated kernel names to show (default: all)")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU names to show (default: all)")
    parser.add_argument("--metric", default=None, choices=("speed", "memory"), help="Filter by metric (verbose only)")
    parser.add_argument("--verbose", action="store_true", help="Show full per-x breakdown instead of summary")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_rows(path)
    filter_kernels = set(args.kernels.split(",")) if args.kernels else None
    filter_gpus = set(args.gpus.split(",")) if args.gpus else None

    print(f"\nFile : {path}  ({len(rows)} rows)")

    if args.verbose:
        print("cuTile speedup = liger p50 / cuTile p50  (> 1 means cuTile is faster / uses less memory)")
        tables, meta = build_verbose_tables(
            rows,
            filter_kernels=filter_kernels,
            filter_metric=args.metric,
            filter_gpus=filter_gpus,
        )
        for gpu in sorted(tables):
            print(f"\n{_BOLD}{_CYAN}GPU: {gpu}{_RESET}")
            for kernel in sorted(tables[gpu]):
                for metric in [m for m in METRIC_ORDER if m in tables[gpu][kernel]]:
                    x_name, unit = meta[(gpu, kernel, metric)]
                    print_verbose_table(kernel, metric, unit, x_name, tables[gpu][kernel][metric])
    else:
        stats = build_kernel_stats(rows, filter_kernels=filter_kernels, filter_gpus=filter_gpus)
        for gpu in sorted(stats):
            print_summary_table(stats[gpu], gpu_name=gpu)

    print()


if __name__ == "__main__":
    main()
