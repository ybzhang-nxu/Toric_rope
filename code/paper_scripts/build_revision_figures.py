from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False


PROJECT_DIR = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_DIR / "paper"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"

BLUE = "#2f6f9f"
GREEN = "#3f8f5f"
ORANGE = "#c77c2f"
RED = "#a6423a"
PURPLE = "#7357a6"
TEAL = "#0f8f91"
GRAY = "#6e7781"
LIGHT_GRAY = "#d7dce2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pick(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight", pad_inches=0.06, dpi=300)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_v12_sector_intervention() -> None:
    rows = read_csv(
        RESULTS_DIR
        / "v12_e5_sector_intervention_cifar10_top110_30k_3seed"
        / "sector_intervention_results.csv"
    )
    specs = [
        ("full", "all_positional", "Full", GRAY),
        ("keep", "axial_J0", "Axial component", BLUE),
        ("keep", "oblique_J0", "Oblique component", GREEN),
        ("zero_bias", "none", "Zero bias", RED),
    ]

    fig, ax = plt.subplots(figsize=(5.7, 2.9))
    y = np.arange(len(specs))
    for idx, (mode, sector, label, color) in enumerate(specs):
        vals = [
            f(row, "score")
            for row in rows
            if row["eval_mode"] == mode and row["sector"] == sector
        ]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        ax.barh(idx, mean, color=color, alpha=0.22, edgecolor=color, height=0.58)
        ax.errorbar(
            mean,
            idx,
            xerr=std,
            fmt="o",
            color=color,
            markersize=5,
            capsize=3,
            zorder=3,
        )
        jitter = np.linspace(-0.11, 0.11, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(vals, idx + jitter, s=20, color=color, alpha=0.52, zorder=2)
        ax.text(mean + 0.018, idx, f"{mean:.3f}", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y, [label for _, _, label, _ in specs])
    ax.invert_yaxis()
    ax.set_xlabel(r"Masked reconstruction $R^2$")
    ax.set_title("CIFAR J0 sector intervention, 3 seeds")
    ax.set_xlim(0.0, 0.84)
    clean_axis(ax)
    fig.subplots_adjust(left=0.26, right=0.96, top=0.84, bottom=0.20)
    save(fig, "fig_mta02_cifar_sector_intervention.pdf")


def plot_v08_boundary_holdout() -> None:
    raw_rows = read_csv(RESULTS_DIR / "v4_offset_holdout_cifar10_10k" / "offset_holdout_aggregate.csv")
    dct_eval_rows = read_csv(
        RESULTS_DIR
        / "v4_offset_holdout_eval_controls_dct_h1b10_cifar10_10k"
        / "offset_holdout_eval_controls_aggregate.csv"
    )
    dct_reg_rows = read_csv(
        RESULTS_DIR
        / "v4_offset_holdout_eval_controls_dct_h1b10_cifar10_10k"
        / "offset_holdout_aggregate.csv"
    )

    raw_specs = [
        ("Table\nlookup", "relative_2d_table", GRAY),
        ("DCT\nraw", "dct_top33", ORANGE),
        ("Toric\nraw", "table_informed_toric_PJ_R0_top110", BLUE),
        ("Axis+PJ\nraw", "axis_plus_toric_residual_R0_top55", TEAL),
    ]
    policy_specs = [
        ("Raw", "raw", ""),
        ("Reg.", "reg", ""),
        ("Clamp", "heldout_clamp", ""),
        ("Decay\n$\\gamma$=1", "radial_decay", "gamma=1"),
        ("Zero-fill", "visible_only", ""),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.75, 3.05), gridspec_kw={"width_ratios": [1.05, 1.05]})

    ax = axes[0]
    x = np.arange(len(raw_specs))
    raw_scores = [f(pick(raw_rows, basis=basis), "final_score_mean") for _, basis, _ in raw_specs]
    raw_colors = [color for _, _, color in raw_specs]
    bars = ax.bar(x, raw_scores, color=raw_colors, alpha=0.88)
    for xpos, val in zip(x, raw_scores):
        va = "bottom" if val >= 0 else "top"
        dy = 0.025 if val >= 0 else -0.025
        ax.text(xpos, val + dy, f"{val:.3f}", ha="center", va=va, fontsize=7)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Raw functional continuation")
    ax.set_ylabel(r"Full-task $R^2$")
    ax.set_xticks(x, [label for label, _, _ in raw_specs])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(-0.16, 0.90)
    clean_axis(ax)

    ax = axes[1]
    raw_dct = pick(raw_rows, basis="dct_top33")
    reg_dct = dct_reg_rows[0]
    values = []
    for label, mode, param in policy_specs:
        if mode == "raw":
            values.append(f(raw_dct, "final_score_mean"))
        elif mode == "reg":
            values.append(f(reg_dct, "final_score_mean"))
        else:
            row = pick(dct_eval_rows, eval_mode=mode, eval_param=param)
            values.append(f(row, "score_mean"))
    x = np.arange(len(policy_specs))
    bars = ax.bar(x, values, color=[ORANGE, ORANGE, GRAY, PURPLE, TEAL], alpha=0.88)
    for xpos, val in zip(x, values):
        va = "bottom" if val >= 0 else "top"
        dy = 0.025 if val >= 0 else -0.025
        ax.text(xpos, val + dy, f"{val:.3f}", ha="center", va=va, fontsize=7)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("DCT boundary-policy sweep")
    ax.set_ylabel(r"Full-task $R^2$")
    ax.set_xticks(x, [label for label, _, _ in policy_specs])
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_ylim(-0.16, 0.86)
    clean_axis(ax)

    fig.suptitle("MT-A08 masked-offset deployment diagnostics", y=0.98, fontsize=10.5)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.79, bottom=0.22, wspace=0.32)
    save(fig, "fig_mta08_boundary_holdout.pdf", tight=False)


def canonical_freq_key(kt: int, kf: int) -> tuple[int, int]:
    return min((kt, kf), (-kt, -kf))


def top_fft_omegas(table: np.ndarray, k: int) -> list[np.ndarray]:
    centered = table - table.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    power[0, 0] = 0.0
    height, width = table.shape
    order = np.argsort(power.reshape(-1))[::-1]
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in order:
        kt = int(item // width)
        kf = int(item % width)
        signed_t = kt if kt <= height // 2 else kt - height
        signed_f = kf if kf <= width // 2 else kf - width
        if signed_t == 0 and signed_f == 0:
            continue
        key = canonical_freq_key(signed_t, signed_f)
        if key in seen:
            continue
        seen.add(key)
        selected.append((signed_t, signed_f))
        if len(selected) >= k:
            break
    return [
        np.array(
            [2.0 * math.pi * signed_t / float(height), 2.0 * math.pi * signed_f / float(width)]
        )
        for signed_t, signed_f in selected
    ]


def full_jet_design(d: np.ndarray, omegas: list[np.ndarray], order: int, time_tokens: int, freq_tokens: int) -> np.ndarray:
    tau = d[:, 0] / float(max(time_tokens - 1, 1))
    phi = d[:, 1] / float(max(freq_tokens - 1, 1))
    cols = [np.ones(d.shape[0], dtype=np.float64)]
    for omega in omegas:
        phase = d @ omega
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        for total in range(order + 1):
            for rt in range(total + 1):
                rf = total - rt
                poly = np.ones_like(cos_phase) if total == 0 else (tau**rt) * (phi**rf)
                cols.extend([poly * cos_phase, poly * sin_phase])
    return np.stack(cols, axis=1)


def fit_pred(x: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=0)
    norms = np.maximum(norms, 1e-12)
    xn = x / norms.reshape(1, -1)
    coeff = np.linalg.solve(xn.T @ xn + ridge * np.eye(xn.shape[1]), xn.T @ y)
    return xn @ coeff


def plot_v13() -> None:
    rows = read_csv(RESULTS_DIR / "v13_time_frequency_jet_positive" / "positive_margins.csv")
    selected = [
        ("j1_time_envelope", "full_j1_minus_j0_cluster", "J1 time\nenvelope"),
        ("j2_directional_packet", "full_j2_minus_j0_cluster", "J2 directional\npacket"),
        ("j2_mixed_envelope", "full_j2_minus_j0_cluster", "J2 mixed\nenvelope"),
        ("mix_j0_j1_j2", "full_j2_minus_j0_cluster", "J0/J1/J2\nmixture"),
    ]

    labels = []
    full = []
    ext = []
    for target, comparison, label in selected:
        row = pick(rows, target=target, noise="0.0", comparison=comparison)
        labels.append(label)
        full.append(f(row, "r2_full_margin"))
        ext.append(f(row, "r2_ext_margin"))

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.bar(x - width / 2, full, width, color=BLUE, label="Full window")
    ax.bar(x + width / 2, ext, width, color=ORANGE, label="Extended grid")
    for xpos, value in zip(x - width / 2, full):
        ax.text(xpos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    for xpos, value in zip(x + width / 2, ext):
        ax.text(xpos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("MT-A04 controlled high-order toric PJ recovery")
    ax.set_ylabel(r"$R^2$ margin over J0 cluster")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(ext) + 0.12)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    clean_axis(ax)
    save(fig, "fig_mta04_time_frequency_margins.pdf")


def plot_v15() -> None:
    result_dir = RESULTS_DIR / "v15_nsynth_cqt_table_projection_256_top6"
    table = np.load(result_dir / "offset_table.npy").astype(np.float64)
    summary_rows = read_csv(result_dir / "projection_results.csv")
    time_tokens = 32
    freq_tokens = 14
    dt = np.arange(-(time_tokens - 1), time_tokens, dtype=np.float64)
    df = np.arange(-(freq_tokens - 1), freq_tokens, dtype=np.float64)
    t_grid, f_grid = np.meshgrid(dt, df, indexing="ij")
    d = np.stack([t_grid.reshape(-1), f_grid.reshape(-1)], axis=1)
    y = table.reshape(-1)
    omegas = top_fft_omegas(table, k=6)

    preds = {}
    r2s = {}
    for order in [0, 1, 2]:
        x = full_jet_design(d, omegas, order, time_tokens, freq_tokens)
        pred = fit_pred(x, y).reshape(table.shape)
        preds[order] = pred
        r2s[order] = f(pick(summary_rows, basis=f"fft_top6_J{order}"), "r2")

    recon_values = [table, preds[0], preds[1], preds[2]]
    residual_values = [np.zeros_like(table), table - preds[0], table - preds[1], table - preds[2]]
    recon_lim = float(np.max(np.abs(np.stack(recon_values))))
    resid_lim = float(np.max(np.abs(np.stack(residual_values[1:]))))

    fig, axes = plt.subplots(2, 4, figsize=(7.75, 4.08))
    top_titles = ["Observed", f"J0 fit\n$R^2$={r2s[0]:.3f}", f"J1 fit\n$R^2$={r2s[1]:.3f}", f"J2 fit\n$R^2$={r2s[2]:.3f}"]
    for idx, arr in enumerate(recon_values):
        ax = axes[0, idx]
        im = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-recon_lim, vmax=recon_lim, aspect="auto")
        ax.set_title(top_titles[idx], fontsize=9)
        ax.set_xticks([0, 13, 26], labels=["-13", "0", "13"], fontsize=6)
        ax.set_yticks([0, 31, 62], labels=["-31", "0", "31"], fontsize=6)
        if idx == 0:
            ax.set_ylabel(r"$\Delta t$", fontsize=8)
        else:
            ax.set_yticklabels([])
    for idx, arr in enumerate(residual_values):
        ax = axes[1, idx]
        if idx == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, "Residuals\nuse shared\nzero-centered scale", ha="center", va="center", fontsize=8)
            continue
        im_res = ax.imshow(arr, origin="lower", cmap="coolwarm", vmin=-resid_lim, vmax=resid_lim, aspect="auto")
        ax.set_title(f"Observed - J{idx - 1}", fontsize=9)
        ax.set_xticks([0, 13, 26], labels=["-13", "0", "13"], fontsize=6)
        ax.set_yticks([0, 31, 62], labels=["-31", "0", "31"], fontsize=6)
        if idx > 1:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$\Delta f$", fontsize=8)
    # Use explicit colorbar axes so the bars do not collide with the rightmost
    # panel titles after LaTeX scales the PDF.
    cax_top = fig.add_axes([0.905, 0.565, 0.016, 0.255])
    cax_bottom = fig.add_axes([0.905, 0.190, 0.016, 0.255])
    fig.colorbar(im, cax=cax_top)
    fig.colorbar(im_res, cax=cax_bottom)
    fig.suptitle("MT-A05 NSynth/CQT empirical offset field and nested Toric PJ fits", y=0.96, fontsize=11)
    fig.subplots_adjust(left=0.050, right=0.885, top=0.82, bottom=0.09, wspace=0.12, hspace=0.42)
    save(fig, "fig_mta05_nsynth_empirical_projection.pdf", tight=False)


def plot_v16() -> None:
    rows = read_csv(
        RESULTS_DIR
        / "v16_nsynth_cqt_projection_stability_256_subsets"
        / "projection_stability_margin_aggregate.csv"
    )
    comparisons = [("J1_minus_J0", "J1-J0"), ("J2_minus_J1", "J2-J1")]
    subsets = ["128", "192", "256"]
    colors = [BLUE, GREEN, ORANGE]
    x = np.arange(len(comparisons))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.15), gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax = axes[0]
    for idx, subset in enumerate(subsets):
        vals = [
            f(pick(rows, subset_size=subset, comparison=comparison), "r2_margin_mean")
            for comparison, _ in comparisons
        ]
        ax.bar(x + (idx - 1) * width, vals, width, color=colors[idx], label=f"n={subset}")
        for xpos, value in zip(x + (idx - 1) * width, vals):
            ax.text(xpos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Incremental hierarchy gains")
    ax.set_ylabel(r"Projection $R^2$ margin")
    ax.set_xticks(x, [label for _, label in comparisons])
    ax.set_ylim(0, 0.10)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean_axis(ax)

    ax = axes[1]
    vals = [f(pick(rows, subset_size=subset, comparison="J2_minus_shuffle"), "r2_margin_mean") for subset in subsets]
    xpos = np.arange(len(subsets))
    ax.bar(xpos, vals, color=colors, alpha=0.95)
    for xitem, value in zip(xpos, vals):
        ax.text(xitem, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Coordinate-shuffle gap")
    ax.set_ylabel(r"Projection $R^2$ margin")
    ax.set_xticks(xpos, [f"n={subset}" for subset in subsets])
    ax.set_ylim(0.0, 1.0)
    clean_axis(ax)
    save(fig, "fig_mta05_nsynth_subset_margins.pdf")


def plot_v17() -> None:
    rows = read_csv(
        RESULTS_DIR
        / "v17_nsynth_learned_bias_projection_3seed_2k"
        / "learned_bias_projection_mean_aggregate.csv"
    )
    case_rows = read_csv(
        RESULTS_DIR
        / "v17_nsynth_learned_bias_projection_3seed_2k"
        / "learned_bias_projection_rows.csv"
    )
    cases: dict[tuple[str, str], dict[str, float]] = {}
    for row in case_rows:
        key = (row["source"], row["table_id"])
        if row["basis"] in {"fft_top6_J0", "fft_top6_J1", "fft_top6_J2"}:
            cases.setdefault(key, {})[row["basis"]] = f(row, "r2")

    labels = ["J0", "J1", "J2"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    all_values = []
    for values in cases.values():
        if all(key in values for key in ["fft_top6_J0", "fft_top6_J1", "fft_top6_J2"]):
            yvals = [values["fft_top6_J0"], values["fft_top6_J1"], values["fft_top6_J2"]]
            all_values.append(yvals)
            ax.plot(x, yvals, color=GRAY, alpha=0.36, linewidth=1.0, marker="o", markersize=2.8)
    median = np.median(np.array(all_values), axis=0)
    ax.plot(x, median, color=ORANGE, linewidth=2.2, marker="o", markersize=4.5, label="Median")
    ax.set_title("MT-A06 learned-bias nested projection cases")
    ax.set_ylabel(r"Projection $R^2$")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.68, 0.99)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "fig_mta06_nsynth_learned_bias_summary.pdf")


def plot_v18() -> None:
    rows = read_csv(
        RESULTS_DIR
        / "v18_nsynth_learned_bias_holdout_3seed_2k"
        / "learned_bias_holdout_margins.csv"
    )
    base_rows = read_csv(
        RESULTS_DIR
        / "v18_nsynth_learned_bias_holdout_3seed_2k"
        / "learned_bias_holdout_rows.csv"
    )
    top18_rows = read_csv(
        RESULTS_DIR
        / "v20_nsynth_matched_j0_control_top18_random"
        / "learned_bias_holdout_rows.csv"
    )
    top36_rows = read_csv(
        RESULTS_DIR
        / "v20_nsynth_matched_j0_control_top36_random"
        / "learned_bias_holdout_rows.csv"
    )

    def row_key(row: dict[str, str], basis: str | None = None) -> tuple[str, str, str, str, str, str]:
        return (
            row["source"],
            row["table_id"],
            row["split_scheme"],
            row["split_seed"],
            row["ridge"],
            basis if basis is not None else row["basis"],
        )

    base_by_key = {
        row_key(row): row
        for row in base_rows
        if row["split_scheme"] == "random" and row["ridge"] == "1e-06"
    }
    top18_by_key = {
        row_key(row): row
        for row in top18_rows
        if row["split_scheme"] == "random" and row["ridge"] == "1e-06"
    }
    top36_by_key = {
        row_key(row): row
        for row in top36_rows
        if row["split_scheme"] == "random" and row["ridge"] == "1e-06"
    }

    groups = sorted(
        {
            (row["source"], row["table_id"], row["split_scheme"], row["split_seed"], row["ridge"])
            for row in base_rows
            if row["split_scheme"] == "random" and row["ridge"] == "1e-06"
        }
    )
    matched_j1: list[float] = []
    matched_j2: list[float] = []
    for source, table_id, scheme, split_seed, ridge in groups:
        prefix = (source, table_id, scheme, split_seed, ridge)
        j1 = f(base_by_key[prefix + ("fft_top6_J1",)], "heldout_r2")
        j2 = f(base_by_key[prefix + ("fft_top6_J2",)], "heldout_r2")
        j0_18 = f(top18_by_key[prefix + ("fft_top18_J0",)], "heldout_r2")
        j0_36 = f(top36_by_key[prefix + ("fft_top36_J0",)], "heldout_r2")
        matched_j1.append(j1 - j0_18)
        matched_j2.append(j2 - j0_36)

    nested_values = []
    for comparison in ["J1_minus_J0", "J2_minus_J1"]:
        nested_values.append(
            [
                f(row, "heldout_margin")
                for row in rows
                if row["split_scheme"] == "random" and row["ridge"] == "1e-06" and row["comparison"] == comparison
            ]
        )
    shuffle_values = [
        f(row, "heldout_margin")
        for row in rows
        if row["split_scheme"] == "random" and row["ridge"] == "1e-06" and row["comparison"] == "J2_minus_shuffle"
    ]

    summary_rows = []
    for label, vals in [
        ("nested_J1_minus_J0", nested_values[0]),
        ("nested_J2_minus_J1", nested_values[1]),
        ("matched_J1top6_minus_J0top18", matched_j1),
        ("matched_J2top6_minus_J0top36", matched_j2),
        ("J2_minus_shuffled_J2", shuffle_values),
    ]:
        arr = np.array(vals, dtype=np.float64)
        summary_rows.append(
            {
                "comparison": label,
                "n_table_mask": len(vals),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "positive": int(np.sum(arr > 0.0)),
            }
        )
    write_csv(RESULTS_DIR / "v20_nsynth_matched_j0_control_summary.csv", summary_rows)

    def seed_from_source(source: str) -> str:
        marker = "relative_2d_table_seed"
        start = source.rfind(marker)
        if start < 0:
            marker = "seed"
            start = source.rfind(marker)
        if start < 0:
            return "unknown"
        start += len(marker)
        end = start
        while end < len(source) and source[end].isdigit():
            end += 1
        return source[start:end] or "unknown"

    value_sets: dict[str, list[tuple[str, str, float]]] = {
        "nested_J1_minus_J0": [],
        "nested_J2_minus_J1": [],
        "J2_minus_shuffled_J2": [],
        "matched_J1top6_minus_J0top18": [],
        "matched_J2top6_minus_J0top36": [],
    }
    for row in rows:
        if row["split_scheme"] == "random" and row["ridge"] == "1e-06":
            label = {
                "J1_minus_J0": "nested_J1_minus_J0",
                "J2_minus_J1": "nested_J2_minus_J1",
                "J2_minus_shuffle": "J2_minus_shuffled_J2",
            }.get(row["comparison"])
            if label is not None:
                value_sets[label].append(
                    (seed_from_source(row["source"]), row["table_id"], f(row, "heldout_margin"))
                )
    for (source, table_id, _, _, _), v1, v2 in zip(groups, matched_j1, matched_j2):
        seed = seed_from_source(source)
        value_sets["matched_J1top6_minus_J0top18"].append((seed, table_id, v1))
        value_sets["matched_J2top6_minus_J0top36"].append((seed, table_id, v2))

    clustered_rows = []
    for label, vals in value_sets.items():
        by_case: dict[tuple[str, str], list[float]] = {}
        by_seed: dict[str, list[float]] = {}
        for seed, table_id, value in vals:
            by_case.setdefault((seed, table_id), []).append(value)
            by_seed.setdefault(seed, []).append(value)
        case_means = [float(np.mean(case_vals)) for case_vals in by_case.values()]
        seed_means = [float(np.mean(seed_vals)) for seed_vals in by_seed.values()]
        clustered_rows.append(
            {
                "comparison": label,
                "n_table_mask": len(vals),
                "table_mask_mean": float(np.mean([v for _, _, v in vals])),
                "n_case_mean": len(case_means),
                "case_positive_count": int(np.sum(np.array(case_means) > 0.0)),
                "case_min": float(np.min(case_means)),
                "n_seed": len(seed_means),
                "seed_mean": float(np.mean(seed_means)),
                "seed_min": float(np.min(seed_means)),
                "seed_max": float(np.max(seed_means)),
            }
        )
    write_csv(RESULTS_DIR / "v20_nsynth_matched_j0_control_seed_clustered.csv", clustered_rows)

    def draw_panel(
        ax: plt.Axes,
        values_by_label: list[list[float]],
        labels: list[str],
        colors: list[str],
        title: str,
        ylim: tuple[float, float],
    ) -> None:
        x = np.arange(len(labels))
        box = ax.boxplot(
            values_by_label,
            positions=x,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
        for idx, vals in enumerate(values_by_label):
            jitter = np.linspace(-0.17, 0.17, len(vals))
            ax.scatter(np.full(len(vals), x[idx]) + jitter, vals, s=10, color=colors[idx], alpha=0.55)
            ax.text(
                x[idx],
                max(vals) + 0.035 * (ylim[1] - ylim[0]),
                f"{sum(v > 0 for v in vals)}/{len(vals)} evals > 0",
                ha="center",
                fontsize=6.5,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        clean_axis(ax)

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 2.95), gridspec_kw={"width_ratios": [1.0, 1.0, 0.75]})
    draw_panel(
        axes[0],
        nested_values,
        ["J1-J0", "J2-J1"],
        [GREEN, ORANGE],
        "Nested increments",
        (-0.01, 0.16),
    )
    axes[0].set_ylabel(r"Heldout $R^2$ margin")
    draw_panel(
        axes[1],
        [matched_j1, matched_j2],
        ["J1 - J0-18", "J2 - J0-36"],
        [BLUE, PURPLE],
        "Matched J0 controls",
        (-0.075, 0.085),
    )
    draw_panel(
        axes[2],
        [shuffle_values],
        ["J2 -\nshuffled J2"],
        [PURPLE],
        "Shuffle gap",
        (0.82, 1.04),
    )
    fig.suptitle(r"MT-A06 random in-window table-mask margins, $\lambda=10^{-6}$", y=0.98, fontsize=10.5)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.80, bottom=0.22, wspace=0.42)
    save(fig, "fig_mta06_nsynth_random_holdout_margins.pdf", tight=False)


def main() -> None:
    plot_v12_sector_intervention()
    plot_v08_boundary_holdout()
    plot_v13()
    plot_v15()
    plot_v16()
    plot_v17()
    plot_v18()


if __name__ == "__main__":
    main()
