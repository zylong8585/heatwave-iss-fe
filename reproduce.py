"""Reproduce the heatwave mean-trajectory results of condition-weighted FE.

Default: evaluate the bundled FE checkpoints and check the metrics against
``reference_metrics.json``.  ``--train`` refits the FE bases from the bundled
initialization (``--epochs`` and ``--seeds`` allow a short smoke test).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import fe

ROOT = Path(__file__).resolve().parent
EXAMPLE_DATE, EXAMPLE_SEED = "2023-07-15", 31626      # day shown in Fig. 5 of the paper
COLORS = {"truth": "#404040", "base": "#2166AC", "prefix_joint": "#9DBBD5", "balanced_prefix": "#B2182B"}
STYLE = {"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02}


# ----------------------------------------------------------------------------- checks
def verify_inputs() -> None:
    """Check the SHA-256 digests of all bundled data, checkpoints, and settings."""
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")


def check_reference(per_run: pd.DataFrame, refit: bool = False) -> dict[str, float]:
    """Compare per-run metrics with the bundled reference values.

    Evaluating the bundled checkpoints reproduces the reference to floating-point
    precision. Refitting is not bit-identical across platforms and thread counts,
    so ``refit=True`` uses a tolerance of 0.05 pp (SoC) and 0.005 degC.
    """
    reference = pd.DataFrame(json.loads((ROOT / "reference_metrics.json").read_text(encoding="utf-8")))
    keys = ["method", "seed", "year", "group"]
    reference["seed"] = reference["seed"].fillna(-1).astype(int)      # M2 has no fitting seed
    actual = per_run.copy()
    actual["seed"] = actual["seed"].fillna(-1).astype(int)
    joined = actual.merge(reference, on=keys, suffixes=("", "_reference"), validate="one_to_one")
    if len(joined) != len(reference):
        raise AssertionError("Missing comparison results")
    deviations = {}
    tolerances = (("soc_mae_pp", 5e-2), ("temperature_mae_C", 5e-3)) if refit else \
        (("soc_mae_pp", 5e-4), ("temperature_mae_C", 5e-5))
    for metric, tolerance in tolerances:
        deviations[metric] = float(abs(joined[metric] - joined[metric + "_reference"]).max())
        if deviations[metric] > tolerance:
            raise AssertionError(f"Reference mismatch: {metric}, max difference={deviations[metric]:.3g}")
    return deviations


# ----------------------------------------------------------------------------- fitting and scoring
def fit(study, method: str, seed: int, tasks: list, epochs: int, output: Path) -> fe.JointFE:
    """Refit the shared FE bases with AdamW (Algorithm 1, mean-training loop)."""
    torch.manual_seed(seed)
    model = fe.model_for(study, initial=True)
    cfg = study.cfg
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    target = fe.training_targets(study.train, study.device, cfg)
    day_weights = torch.tensor(fe.sample_weights(study.train), device=study.device, dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        total = 0.0
        model.train()
        for fold in np.random.default_rng(seed + 1009 * epoch).permutation(len(tasks)):
            qi, si, q, source, sw, plan, logits = tasks[fold]
            optimizer.zero_grad(set_to_none=True)
            delta, _ = fe.forward_correction(model, q, source, target["residual"][si], sw, plan, method, cfg, logits)
            loss = (fe.loss_per_day(delta, fe.subset(target, qi), cfg) * day_weights[qi]).sum() * cfg["folds"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) / len(tasks)
        history.append({"epoch": epoch + 1, "loss": total})
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            print(f"  {fe.LABELS[method]}, seed {seed}: epoch {epoch + 1}/{epochs}", flush=True)
    folder = output / "checkpoints"
    folder.mkdir(exist_ok=True)
    tag = f"{method}_s{seed}"
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "method": method, "seed": seed, "epochs": epochs}, folder / f"{tag}.pt")
    pd.DataFrame(history).to_csv(folder / f"{tag}_history.csv", index=False)
    return model


def score(study, prediction: np.ndarray, method: str, seed: int | None) -> pd.DataFrame:
    """Daily MAE of SoC (percentage points) and zone temperature (degC) over hours 1-24."""
    error = abs(prediction - study.dev["truth"][:, 1:])
    return pd.DataFrame({"method": method, "seed": seed, "day_id": study.dev["ids"], "year": study.dev["year"],
                         "group": np.where(study.dev["hot"], "heatwave", "normal"),
                         "soc_mae_pp": error[..., 4].mean(axis=1) * 100,
                         "temperature_mae_C": error[..., 3].mean(axis=1)})


def largest_window_weight(study, day_weights: np.ndarray) -> np.ndarray:
    """Largest total weight assigned to one historical window, per query day and hour."""
    plan = fe.window_plan(study.train, study.device)
    group = plan["inverse"].cpu().numpy()
    mass = np.stack([day_weights[..., group == j].sum(-1) for j in range(len(plan["episodes"]))], axis=-1)
    return mass.max(-1)


# ----------------------------------------------------------------------------- reporting
def markdown_table(summary: pd.DataFrame) -> str:
    """Table II-style summary on heatwave days (mean +/- std over fitting runs)."""
    lines = ["| Model | 2022 SoC (pp) | 2022 $T_z$ (°C) | 2023 SoC (pp) | 2023 $T_z$ (°C) |",
             "|---|---:|---:|---:|---:|"]
    for method in ("base", *fe.METHODS):
        bold = "**" if method == "balanced_prefix" else ""
        cells = []
        for year in (2022, 2023):
            row = summary[(summary["method"] == method) & (summary["year"] == year)
                          & (summary["group"] == "heatwave")].iloc[0]
            for metric in ("soc_mae_pp", "temperature_mae_C"):
                cell = f"{bold}{row[metric]:.4f}{bold}"
                if method != "base":
                    cell += f" ± {row[metric + '_std']:.4f}"
                cells.append(cell)
        lines.append(f"| {bold}{fe.LABELS[method]}{bold} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_figures(study, summary: pd.DataFrame, weights: dict, example: dict, output: Path) -> None:
    plt.rcParams.update(STYLE)

    # (1) Heatwave-day MAE (mean and std over fitting runs)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), layout="constrained")
    methods = ("base", *fe.METHODS)
    for ax, metric, ylabel in zip(axes, ("soc_mae_pp", "temperature_mae_C"),
                                  ("SoC MAE (pp)", r"$T_{\mathrm{z}}$ MAE ($^\circ$C)")):
        for offset, method in enumerate(methods):
            rows = summary[(summary["method"] == method) & (summary["group"] == "heatwave")].sort_values("year")
            spread = None if method == "base" else rows[metric + "_std"]     # M2 has no fitting runs
            ax.bar(np.arange(2) + (offset - 1) * 0.25, rows[metric], width=0.24, yerr=spread,
                   color=COLORS[method], label=fe.LABELS[method], capsize=2, error_kw={"lw": 0.7})
        ax.set_xticks([0, 1], ["2022 heatwave days", "2023 heatwave days"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"heatwave_mae.{suffix}", dpi=220)
    plt.close(fig)

    # (2) Example heatwave day (paper Fig. 5 day): simulation, M2, proposed
    hours = np.arange(25)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True, layout="constrained")
    for ax, k, scale, ylabel in ((axes[0], 4, 100, "SoC (%)"), (axes[1], 3, 1, r"Zone temperature ($^\circ$C)")):
        ax.plot(hours, example["truth"][:, k] * scale, color=COLORS["truth"], lw=1.3, label="Simulation")
        for key, style in (("base", "--"), ("balanced_prefix", "-")):
            path = np.concatenate([example["truth"][:1, k], example[key][:, k]]) * scale
            ax.plot(hours, path, style, color=COLORS[key], lw=1.1, label=fe.LABELS[key])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].set_xlabel("Elapsed time (h)")
    axes[1].set_xticks(range(0, 25, 6))
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"example_day.{suffix}", dpi=220)
    plt.close(fig)

    # (3) Largest historical-window weight on heatwave query hours
    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout="constrained")
    for method, mass in weights.items():
        values = np.sort(mass[study.dev["hot"]].ravel())
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=COLORS[method],
                lw=1.2, label=fe.LABELS[method])
    ax.axvline(study.cfg["window_cap"], color="#7F8790", ls=":", lw=1, label=r"Bound $w_{\max}=0.25$")
    ax.set(xlabel="Largest historical-window weight", ylabel="Cumulative share of query hours",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"window_weights.{suffix}", dpi=220)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(args: argparse.Namespace) -> None:
    began = time.perf_counter()
    default = "retrained" if args.train else "pretrained"
    output = Path(args.output).resolve() if args.output else ROOT / "results" / default
    protected = [ROOT / "data", ROOT / "checkpoints"]
    if output == ROOT or output in ROOT.parents or any(output == p or p in output.parents for p in protected):
        raise ValueError("Choose a separate output directory")
    verify_inputs()
    study = fe.load_study(ROOT, args.device)
    cfg = study.cfg
    seeds = args.seeds or cfg["seeds"]
    epochs = args.epochs or cfg["epochs"]
    full_run = set(seeds) == set(cfg["seeds"]) and (not args.train or epochs == cfg["epochs"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    dates = pd.read_csv(ROOT / "data/days.csv")[["id", "date"]].rename(columns={"id": "day_id"})
    example_ix = int(np.flatnonzero(study.dev["ids"] == int(dates.loc[dates["date"] == EXAMPLE_DATE, "day_id"].iloc[0]))[0])
    example = {"truth": study.dev["truth"][example_ix], "base": study.dev["base"][example_ix]}

    tasks = fe.train_tasks(study) if args.train else None
    frames = [score(study, study.dev["base"], "base", None)]
    weights = {}
    for method in fe.METHODS:
        for seed in seeds:
            print(f"{'Refitting' if args.train else 'Evaluating'} {fe.LABELS[method]}, seed {seed}", flush=True)
            model = fit(study, method, seed, tasks, epochs, output) if args.train else fe.model_for(study, method, seed)
            first = seed == seeds[0]
            result = fe.predict(study, model, study.dev, method, return_weights=first)
            prediction = result[0] if first else result
            if first:
                weights[method] = largest_window_weight(study, result[1])
            if seed == EXAMPLE_SEED or (first and EXAMPLE_SEED not in seeds):
                example[method] = prediction[example_ix]
            np.savez_compressed(output / "predictions" / f"{method}_s{seed}.npz", mean=prediction,
                                day_ids=study.dev["ids"])
            frames.append(score(study, prediction, method, seed))

    daily = pd.concat(frames, ignore_index=True)
    daily.merge(dates, on="day_id", validate="many_to_one").to_csv(output / "daily_metrics.csv", index=False)
    metrics = ["soc_mae_pp", "temperature_mae_C"]
    per_run = daily.groupby(["method", "seed", "year", "group"], dropna=False)[metrics].mean().reset_index()
    per_run.to_csv(output / "run_metrics.csv", index=False)
    summary = per_run.groupby(["method", "year", "group"])[metrics].agg(["mean", "std"])
    summary.columns = [name if stat == "mean" else name + "_std" for name, stat in summary.columns]
    summary = summary.fillna(0).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    table = markdown_table(summary)
    (output / "table_heatwave.md").write_text(table, encoding="utf-8")
    np.savez_compressed(output / "window_weights.npz", day_ids=study.dev["ids"], **weights)
    save_figures(study, summary, weights, example, output)

    if full_run:
        deviations = check_reference(per_run, refit=args.train)
        status = "passed"
    else:
        deviations, status = {}, "skipped (short or partial run)"
    receipt = {"mode": "refit FE bases" if args.train else "bundled FE checkpoints", "epochs": epochs,
               "seeds": seeds, "device": str(study.device), "python": platform.python_version(),
               "torch": torch.__version__, "numpy": np.__version__, "reference_check": status,
               "maximum_metric_difference": deviations, "elapsed_seconds": round(time.perf_counter() - began, 1)}
    (output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\nHeatwave-day MAE (mean ± std over fitting runs):\n" + table)
    print(f"Reference check: {status}. Results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="refit the FE bases from the bundled initialization")
    parser.add_argument("--epochs", type=int, help="training epochs for --train (default: config value, 120)")
    parser.add_argument("--seeds", type=int, nargs="+", help="fitting seeds (default: 31626 31627 31628)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", help="output directory (default: results/pretrained or results/retrained)")
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run(arguments)
