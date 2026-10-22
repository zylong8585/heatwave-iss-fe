"""Condition-weighted function encoding (FE) for heatwave mean-trajectory adaptation.

This module implements the mean-adaptation component of the paper
(Section III-C): shared learned basis functions, condition-similarity weights,
the bound on each historical window's total weight, and the weighted ridge
solve for the encoding coefficients. The pretrained neural ODE is not
retrained; its trajectories are supplied as fixed inputs (``base``).

Method keys used in file names:
    ``prefix_joint``     condition-weighted FE without the window bound
    ``balanced_prefix``  condition-weighted FE with the window bound (proposed)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

METHODS = ("prefix_joint", "balanced_prefix")
LABELS = {
    "base": "M2 (pretrained neural ODE)",
    "prefix_joint": "FE without window bound",
    "balanced_prefix": "Proposed (FE with window bound)",
}
FEATURE_DIM = 411          # 385 masked input-prefix features + 26 local features
HOURS = 24
LOW_SOC = 0.15             # low-storage threshold (fraction)


# ----------------------------------------------------------------------------- I/O
def arrays(path: Path) -> dict[str, np.ndarray]:
    """Load an ``.npz`` file without pickle support."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def loadpt(path: Path, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint with ``weights_only=True`` (tensors and plain containers only)."""
    return torch.load(path, map_location=device, weights_only=True)


def subset(bundle: dict, index) -> dict:
    """Select the same rows from every array (or tensor) of a bundle."""
    return {key: value[index] for key, value in bundle.items()}


def load_study(root: Path, device: str = "cpu") -> SimpleNamespace:
    """Load the configuration, data, and training folds of the demo."""
    root = Path(root).resolve()
    return SimpleNamespace(
        root=root,
        cfg=json.loads((root / "config.json").read_text(encoding="utf-8")),
        device=torch.device(device),
        train=arrays(root / "data/train.npz"),
        dev=arrays(root / "data/dev.npz"),
        folds=np.load(root / "data/training_year_folds.npy", allow_pickle=False),
    )


# ----------------------------------------------------------------------------- basis functions
def mlp(nin: int, width: int, nout: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(nin, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                         nn.Linear(width, nout))


class FunctionBasis(nn.Module):
    """A constant basis plus ``rank`` learned basis functions psi(t)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.rank = cfg["rank"]
        self.register_buffer("feature_mean", torch.zeros(FEATURE_DIM))
        self.register_buffer("feature_std", torch.ones(FEATURE_DIM))
        self.net = mlp(FEATURE_DIM, cfg["width"], self.rank)

    def phi(self, f: dict) -> torch.Tensor:
        v = torch.tanh(self.net((f["full"] - self.feature_mean) / self.feature_std)) / np.sqrt(self.rank)
        return torch.cat([torch.ones_like(v[..., :1]), v], dim=-1)


class JointFE(nn.Module):
    """Concatenated bases (``pool`` and ``hot`` are checkpoint attribute names)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.pool = FunctionBasis(cfg)
        self.hot = FunctionBasis(cfg)

    def phi(self, f: dict) -> torch.Tensor:
        return torch.cat([self.pool.phi(f), self.hot.phi(f)], -1)


# ----------------------------------------------------------------------------- features
def input_tensors(bundle: dict, state_scale, device) -> dict[str, torch.Tensor]:
    """Use only quantities known before the day starts; ``truth`` is never read here."""
    result = {k: torch.as_tensor(bundle[k], dtype=torch.float32, device=device)
              for k in ("past", "u", "d", "cmd", "base")}
    result["base_scaled"] = result["base"] / torch.as_tensor(state_scale, dtype=torch.float32, device=device)
    return result


def features(b: dict) -> dict[str, torch.Tensor]:
    """Masked input prefixes up to each hour (385 dims) and local state features (26 dims)."""
    n = b["past"].shape[0]
    device, dtype = b["past"].device, b["past"].dtype
    lead = torch.arange(1, HOURS + 1, device=device, dtype=dtype)
    mask = torch.tril(torch.ones(HOURS, HOURS, device=device, dtype=dtype))[None, :, :, None]
    blocks = [b["past"][:, None].expand(-1, HOURS, -1) * np.sqrt(0.25 / 25)]
    for key, dim in (("u", 3), ("d", 7), ("cmd", 5)):
        p = b[key][:, None] * mask
        blocks.append(p.reshape(n, HOURS, -1) * torch.sqrt(0.25 / (lead * dim))[None, :, None])
    prefix = torch.cat(blocks, dim=-1)
    start = torch.cat([b["past"][:, -5:][:, None], b["base_scaled"][:, :-1]], dim=1)
    local = torch.cat([start, b["base_scaled"], b["u"], b["d"], b["cmd"],
                       (lead / HOURS)[None, :, None].expand(n, -1, -1)], dim=-1)
    return {"prefix": prefix, "local": local, "full": torch.cat([prefix, local], dim=-1)}


def tensor_features(st: SimpleNamespace, b: dict) -> dict[str, torch.Tensor]:
    return features(input_tensors(b, st.cfg["state_scale"], st.device))


# ----------------------------------------------------------------------------- condition weights
def bandwidths(source: dict) -> torch.Tensor:
    """Median pairwise squared feature distance for each hour (kernel scale)."""
    values = []
    with torch.no_grad():
        z = source["prefix"].transpose(0, 1)
        d = torch.cdist(z, z, compute_mode="donot_use_mm_for_euclid_dist").square()
        for h in range(HOURS):
            positive = d[h][d[h] > 1e-12]
            values.append(positive.median() if positive.numel() else d.new_tensor(1.0))
    return torch.stack(values).clamp_min(1e-10)


def sample_weights(bundle: dict) -> np.ndarray:
    """Prior weights beta_i that balance normal-operation and heatwave samples and windows."""
    w = np.zeros(len(bundle["ids"]), float)
    active = [g for g in (False, True) if np.any(bundle["hot"] == g)]
    for g in active:
        ix = np.flatnonzero(bundle["hot"] == g)
        ep = bundle["episode"][ix]
        unique, counts = np.unique(ep, return_counts=True)
        count = dict(zip(unique, counts))
        w[ix] = [1 / (len(active) * len(unique) * count[e]) for e in ep]
    return w / w.sum()


def prefix_logits(q: dict, sf: dict, day_weight: torch.Tensor, bw: torch.Tensor) -> torch.Tensor:
    """Log of the unnormalized condition weights in (18) for every query hour."""
    distance = torch.cdist(q["prefix"].transpose(0, 1), sf["prefix"].transpose(0, 1),
                           compute_mode="donot_use_mm_for_euclid_dist").square().permute(1, 0, 2)
    return -distance / (2 * bw[None, :, None]) + day_weight.clamp_min(1e-20).log()[None, None, :]


# ----------------------------------------------------------------------------- window bound (19)
def window_plan(source: dict, device) -> dict:
    """Index historical days by operating window D_g."""
    episodes, inverse = np.unique(source["episode"], return_inverse=True)
    lists = [np.flatnonzero(inverse == j) for j in range(len(episodes))]
    width = max(map(len, lists))
    index = np.zeros((len(lists), width), dtype=np.int64)
    valid = np.zeros_like(index, dtype=bool)
    for j, ix in enumerate(lists):
        index[j, :len(ix)] = ix
        valid[j, :len(ix)] = True
    return {"index": torch.as_tensor(index, device=device), "valid": torch.as_tensor(valid, device=device),
            "inverse": torch.as_tensor(inverse, device=device), "episodes": episodes}


def grouped_logits(logits: torch.Tensor, plan: dict) -> tuple[torch.Tensor, torch.Tensor]:
    g = logits[..., plan["index"]].masked_fill(~plan["valid"], -torch.inf)
    alive = torch.isfinite(g).any(-1)
    safe = torch.where(alive[..., None], g, torch.zeros_like(g))
    return torch.logsumexp(safe, -1).masked_fill(~alive, -torch.inf), alive


def capped_window_projection(logits: torch.Tensor, plan: dict, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """KL projection of softmax(logits) onto {sum w = 1, mass of each window <= limit}.

    The solution saturates at most floor(1/limit) windows and preserves relative
    weights within each window. The active set is found iteratively; the final
    expression is differentiable away from the active-set boundaries.
    """
    logg, alive = grouped_logits(logits, plan)
    count = alive.sum(-1)
    if not bool((count > 0).all()):
        raise ValueError("No eligible historical window")
    cap = torch.maximum(logg.new_full(count.shape, limit), count.to(logg.dtype).reciprocal())
    saturated = torch.zeros_like(alive)
    with torch.no_grad():
        for _ in range(int(np.ceil(1 / limit)) + 2):
            remaining = 1 - cap * saturated.sum(-1)
            proposal = torch.softmax(logg.masked_fill(saturated, -torch.inf), -1) * remaining[..., None]
            new = saturated | (proposal > cap[..., None] + 1e-07) & alive
            if torch.equal(new, saturated):
                break
            saturated = new
        else:
            raise RuntimeError("Window projection active set did not converge")
    remaining = 1 - cap * saturated.sum(-1)
    group_mass = torch.where(saturated, cap[..., None],
                             torch.softmax(logg.masked_fill(saturated, -torch.inf), -1) * remaining[..., None])
    log_group_for_day = logg[..., plan["inverse"]]
    finite = torch.isfinite(logits)
    difference = (torch.where(finite, logits, torch.zeros_like(logits))
                  - torch.where(finite, log_group_for_day, torch.zeros_like(logits)))
    within = torch.exp(difference) * finite.to(logits.dtype)
    return within * group_mass[..., plan["inverse"]], cap


# ----------------------------------------------------------------------------- ridge solve (21)
def joint_solve(qphi: torch.Tensor, sphi: torch.Tensor, target: torch.Tensor, w: torch.Tensor,
                ridge: float) -> torch.Tensor:
    """Solve (Phi^T Omega Phi + lambda_B I) B = Phi^T Omega E per query hour and apply B to psi."""
    gram = torch.einsum("bhs,shr,shk->bhrk", w, sphi, sphi)
    rhs = torch.einsum("bhs,shr,sho->bhro", w, sphi, target)
    eye = torch.eye(sphi.shape[-1], device=sphi.device, dtype=sphi.dtype)
    a = torch.linalg.solve(gram + ridge * eye, rhs)
    return torch.einsum("bhr,bhro->bho", qphi, a)


def forward_correction(model: JointFE, q: dict, sf: dict, response: torch.Tensor, day_weight: torch.Tensor,
                       plan: dict, method: str, cfg: dict, logits: torch.Tensor):
    """Weights (with or without the window bound) followed by the weighted ridge solve."""
    if method not in METHODS:
        raise ValueError(method)
    if method == "balanced_prefix":
        w = capped_window_projection(logits, plan, cfg["window_cap"])[0]
    else:
        w = torch.softmax(logits, -1)
    delta = joint_solve(model.phi(q), model.phi(sf), response, w, cfg["fe_ridge"])
    return delta, w


# ----------------------------------------------------------------------------- training targets and loss
def training_targets(bundle: dict, device, cfg: dict) -> dict[str, torch.Tensor]:
    """Scaled residuals of zone temperature and SoC (the two corrected states)."""
    truth = torch.as_tensor(bundle["truth"][:, 1:, 3:5], dtype=torch.float32, device=device)
    base = torch.as_tensor(bundle["base"][:, :, 3:5], dtype=torch.float32, device=device)
    scale = torch.tensor(cfg["output_scale"], dtype=torch.float32, device=device)
    low = torch.as_tensor(bundle["truth"][:, :-1, 4] < LOW_SOC, device=device)
    return {"residual": (truth - base) / scale, "truth": truth, "base": base, "low": low}


def loss_per_day(delta: torch.Tensor, t: dict, cfg: dict) -> torch.Tensor:
    """Normalized trajectory error (extra weight at low storage), daily-minimum SoC error,
    and mean hourly deficit below the low-storage threshold."""
    def smooth(x):
        return torch.sqrt(x.square() + cfg["smooth_abs"] ** 2) - cfg["smooth_abs"]
    err = smooth(delta - t["residual"])
    soc_weight = 1 + cfg["low_weight"] * t["low"].to(err.dtype)
    fit = 0.5 * (err[:, :, 0].mean(1) + (err[:, :, 1] * soc_weight).sum(1) / soc_weight.sum(1))
    soc = t["base"][:, :, 1] + delta[:, :, 1] * cfg["output_scale"][1]
    actual = t["truth"][:, :, 1]
    minimum = smooth((soc.amin(1) - actual.amin(1)) / cfg["output_scale"][1])
    deficit = smooth(((LOW_SOC - soc).relu().mean(1) - (LOW_SOC - actual).relu().mean(1)) / cfg["output_scale"][1])
    return fit + cfg["minimum_weight"] * minimum + cfg["shortfall_weight"] * deficit


def train_tasks(st: SimpleNamespace) -> list[tuple]:
    """Year-grouped folds: coefficients are fitted on other years, the loss on the held-out years."""
    ft = tensor_features(st, st.train)
    tasks = []
    for f in range(st.cfg["folds"]):
        qi = np.flatnonzero(st.folds == f)
        si = np.flatnonzero(st.folds != f)
        q, sf, source = subset(ft, qi), subset(ft, si), subset(st.train, si)
        sw = torch.tensor(sample_weights(source), device=st.device, dtype=torch.float32)
        with torch.no_grad():
            logits = prefix_logits(q, sf, sw, bandwidths(sf) * st.cfg["kernel_length"])
        tasks.append((qi, si, q, sf, sw, window_plan(source, st.device), logits))
    return tasks


# ----------------------------------------------------------------------------- inference
def model_for(st: SimpleNamespace, method: str = "balanced_prefix", seed: int = 31626,
              initial: bool = False, checkpoint: Path | None = None) -> JointFE:
    p = checkpoint or st.root / "checkpoints" / ("initial.pt" if initial else f"{method}_s{seed}.pt")
    model = JointFE(st.cfg).to(st.device)
    model.load_state_dict(loadpt(p, st.device)["state_dict"])
    return model


def predict(st: SimpleNamespace, model: JointFE, query: dict, method: str = "balanced_prefix",
            return_weights: bool = False):
    """Adapted mean trajectories for the query days.

    Only the query inputs (initial history, prescribed inputs, and the pretrained
    trajectory) are read; query responses are never used. Water-side states are
    returned unchanged; zone temperature and SoC are corrected.
    """
    source = st.train
    sf = tensor_features(st, source)
    response = training_targets(source, st.device, st.cfg)["residual"]
    sw = torch.tensor(sample_weights(source), dtype=torch.float32, device=st.device)
    plan = window_plan(source, st.device)
    bw = bandwidths(sf) * st.cfg["kernel_length"]
    scale = torch.tensor(st.cfg["output_scale"], device=st.device)
    predictions, diagnostics = [], []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(query["ids"]), st.cfg["evaluation_batch"]):
            b = subset(query, slice(j, j + st.cfg["evaluation_batch"]))
            q = tensor_features(st, b)
            logits = prefix_logits(q, sf, sw, bw)
            delta, w = forward_correction(model, q, sf, response, sw, plan, method, st.cfg, logits)
            base = torch.tensor(b["base"], device=st.device, dtype=torch.float32)
            y = torch.cat([base[..., :3], base[..., 3:5] + delta * scale], -1)
            if not torch.isfinite(y).all():
                raise FloatingPointError("Nonfinite prediction")
            predictions.append(y.cpu().numpy())
            if return_weights:
                diagnostics.append(w.cpu().numpy())
    result = np.concatenate(predictions)
    return (result, np.concatenate(diagnostics)) if return_weights else result
