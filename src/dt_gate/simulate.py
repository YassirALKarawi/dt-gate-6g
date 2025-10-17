# src/dt_gate/simulate.py
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    SEEDS, EPOCHS, CLASSES, EPS, D_BUDGET_MS, W,
    ETA, GAMMA, ALPHA, ZETA, TAU_MIN, R_MAX,
    M_SMOOTH, SIGMA_S, ENERGY_BASE
)
from .queueing import beta_cantelli, bootstrap_ci
from .ekf import EKFScalar
from .tube import tube_radius_from_nis
from .risk import RiskMLP, isotonic_calibrate, smooth_samples
from .optimizer import raso_solve
from .telemetry import init_classes, attack, plant_epoch


# ----------------------------
# Ablations / Variants
# ----------------------------
ABLATIONS = [
    {"name": "Proposed",            "tube": True,  "cvar": True,  "smooth": True,  "trust_gate": True},
    {"name": "-- Tube MPC",         "tube": False, "cvar": True,  "smooth": True,  "trust_gate": True},
    {"name": "-- CVaR term",        "tube": True,  "cvar": False, "smooth": True,  "trust_gate": True},
    {"name": "-- Smoothing",        "tube": True,  "cvar": True,  "smooth": False, "trust_gate": True},
    {"name": "-- Trust gate τ_min", "tube": True,  "cvar": True,  "smooth": True,  "trust_gate": False},
]


# ----------------------------
# Helpers
# ----------------------------
def build_feat(obs: Dict[str, Tuple[float, float, float, float]],
               util: float,
               atk: Dict[str, float]) -> np.ndarray:
    """Construct a fixed-length feature vector from observations + utilization + attack flags."""
    feats = []
    # order is deterministic over CLASSES
    for k in CLASSES:
        muD, sigD, lmbd, mu = obs[k]
        feats += [muD, sigD, lmbd, mu]
    feats += [util, atk.get("active", 0.0), atk.get("lambda_boost", 0.0),
              atk.get("mu_drop", 0.0), atk.get("jitter", 0.0)]
    return np.array(feats[:10], dtype=float)


# ---------- JSONL intake (optional external telemetry) ----------
def _obs_from_jsonl_entry(entry: dict) -> Dict[str, Tuple[float, float, float, float]]:
    """Map a JSONL entry to our internal obs format: (muD, sigD, lambda, mu) per class."""
    obs = {}
    clmap = entry.get("classes", {})
    for k in CLASSES:
        c = clmap.get(k, {})
        obs[k] = (
            float(c.get("muD", 0.0)),
            float(c.get("sigD", 0.0)),
            float(c.get("lambda", 0.0)),
            float(c.get("mu", 1e-9)),
        )
    return obs


def jsonl_stream(path: Path) -> Iterable[Tuple[Dict, float, float, int]]:
    """
    Yield (obs, energy, util, label) from a JSONL file.
    Expected entry shape:
      {
        "t": 123,
        "classes": { "URLLC": {"muD":..,"sigD":..,"lambda":..,"mu":..}, ... },
        "util": 0.71,
        "attack": {"active":0/1, "lambda_boost":..., "mu_drop":..., "jitter":...}
      }
    label = 1 if attack.active else 0
    energy is computed by the same deterministic formula using the *current* control gain
    (applied later in the loop).
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            obs = _obs_from_jsonl_entry(e)
            util = float(e.get("util", 0.0))
            atk = e.get("attack", {})
            label = 1 if float(atk.get("active", 0.0)) > 0.5 else 0
            # energy depends on gain; we return atk as hint via closure (handled in run_once_jsonl)
            yield obs, util, label, atk


# ----------------------------
# Core runs
# ----------------------------
def run_once(seed: int,
             cfg: dict,
             jsonl_iter: Optional[Iterable[Tuple[Dict, float, int, Dict]]] = None) -> Tuple[pd.DataFrame, object]:
    """Run a single seeded simulation (either internal plant or external JSONL if provided)."""
    np.random.seed(seed)
    random.seed(seed)

    state = init_classes()
    beta = {k: beta_cantelli(EPS[k]) for k in CLASSES}

    ekf = EKFScalar(mu0=10.0, P0=4.0, Q=0.2, R=1.0)
    mlp = RiskMLP(din=10, dh=16, seed=seed)

    last_safe_gain = 0.0
    logs, raw_scores, labels = [], [], []
    nis_hist: list = []

    # optional JSONL iterator
    jsonl_it = iter(jsonl_iter) if jsonl_iter is not None else None

    # ---- warm-up for isotonic ----
    warm = 150
    for _ in range(warm):
        if jsonl_it is not None:
            try:
                obs, util, label_atk, atk = next(jsonl_it)
            except StopIteration:
                # fall back to internal generator upon exhaustion
                jsonl_it = None
                atk = attack(_)
                obs, energy, util, label_atk = plant_epoch(state, last_safe_gain, atk, ENERGY_BASE)
        else:
            atk = attack(_)
            obs, energy, util, label_atk = plant_epoch(state, last_safe_gain, atk, ENERGY_BASE)

        feat = build_feat(obs, util, atk)
        s = mlp.forward(feat)
        raw_scores.append(s)
        labels.append(label_atk)

    calib = isotonic_calibrate(raw_scores, labels)

    # ---- main control loop ----
    for t in range(EPOCHS):
        if jsonl_it is not None:
            try:
                obs, util, label_atk, atk = next(jsonl_it)
            except StopIteration:
                jsonl_it = None
                atk = attack(t)
                obs, energy, util, label_atk = plant_epoch(state, last_safe_gain, atk, ENERGY_BASE)
            # energy depends on current gain (computed later); placeholder here
            energy = None
        else:
            atk = attack(t)
            obs, energy, util, label_atk = plant_epoch(state, last_safe_gain, atk, ENERGY_BASE)

        # EKF trust from URLLC delay mean
        mu_meas = obs["URLLC"][0]
        tau, NIS, muP, PP = ekf.step(mu_meas)
        nis_hist.append(NIS)

        # risk model (calibrated + smoothing)
        feat = build_feat(obs, util, atk)
        s_raw = mlp.forward(feat)
        s_cal = calib.predict([np.clip(s_raw, 0, 1)])[0]
        risks = smooth_samples(s_cal, M_SMOOTH if cfg["smooth"] else 1,
                               SIGMA_S if cfg["smooth"] else 0.0)

        # trust gate
        gate_ok = ((tau >= TAU_MIN) and (np.mean(risks) <= R_MAX)) if cfg["trust_gate"] else True

        # tube tightening from NIS
        nis_p95 = float(np.quantile(nis_hist[-100:] if len(nis_hist) >= 5 else nis_hist, 0.95)) if nis_hist else 0.0
        tube_m = 0.0 if not cfg["tube"] else tube_radius_from_nis(nis_p95, gain=max(0.3, last_safe_gain + 0.3))

        # optimizer
        best = raso_solve(
            obs, beta,
            params={"tube_margin": tube_m, "NIS": NIS, "D_BUDGET_MS": D_BUDGET_MS},
            risks=risks,
            cfg=cfg,
            weights=W, energy_base=ENERGY_BASE,
            alpha=ALPHA, eta=ETA, gamma=GAMMA, zeta=ZETA
        )

        if best["feas"] and gate_ok:
            gain = best["gain"]; last_safe_gain = gain; accepted = 1
        else:
            gain = last_safe_gain; accepted = 0

        # if energy was not given by plant (JSONL path), compute deterministically
        energy_e = ENERGY_BASE * (1 - 0.15 * gain)
        energy_e = max(0.6, energy_e)

        # SLA & L99 estimates per class (analytical surrogate)
        Ls, viol = [], {}
        for k, (muD, sig, _, _) in obs.items():
            mu_e = muD * (1 - 0.25 * gain)
            sig_e = sig * (1 - 0.20 * gain)
            viol[k] = 1 if (mu_e + beta[k] * sig_e) > D_BUDGET_MS[k] else 0
            Ls.append(mu_e + 2.326 * sig_e)
        L99 = float(max(Ls))

        logs.append({
            "t": t, "gain": gain, "accepted": accepted, "tau": tau,
            "risk": float(np.mean(risks)), "L99": L99, "energy": energy_e,
            "viol_url": viol["URLLC"], "label_atk": label_atk, "tube_m": tube_m,
        })

    return pd.DataFrame(logs), calib


def run_all(variant: Optional[str] = None,
            epochs: int = EPOCHS,
            seeds: Iterable[int] = SEEDS,
            jsonl_path: Optional[Path] = None) -> pd.DataFrame:
    """Run one or more variants over seeds; optionally feed from a JSONL file."""
    global EPOCHS
    EPOCHS = epochs

    cfgs = [c for c in ABLATIONS if (variant is None or c["name"] == variant)]
    frames = []

    jsonl_iter = None
    if jsonl_path is not None and jsonl_path.exists():
        jsonl_iter = jsonl_stream(jsonl_path)

    for cfg in cfgs:
        for s in seeds:
            df, _ = run_once(s, cfg, jsonl_iter=jsonl_iter)
            df["seed"] = s
            df["variant"] = cfg["name"]
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


def save_summary_and_csv(df: pd.DataFrame, out_csv: Path = Path("data/outputs/table_main.csv")) -> None:
    """Aggregate metrics per variant and write a summary CSV."""
    rows = []
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        L = sub["L99"].values
        E = sub["energy"].values
        try:
            auc = float(roc_auc_score(sub["label_atk"].values, 1.0 - sub["risk"].values))
        except Exception:
            auc = float("nan")
        Lm, Llo, Lhi = bootstrap_ci(L)
        Em, Elo, Ehi = bootstrap_ci(E)
        rows.append({
            "Method": v,
            "L99_mean": Lm, "L99_lo": Llo, "L99_hi": Lhi,
            "SLA_URLLC_%": 100 * sub["viol_url"].mean(),
            "Energy_J/MB": Em, "E_lo": Elo, "E_hi": Ehi,
            "AUC": auc, "GateAccept_%": 100 * sub["accepted"].mean(),
        })
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {out_csv}")


# ----------------------------
# CLI
# ----------------------------
def cli():
    parser = argparse.ArgumentParser(
        description="DT-Gate 6G (Python simulation) — variants/ablations with optional JSONL telemetry."
    )
    parser.add_argument("--epochs", type=int, default=700,
                        help="Number of control epochs per run (default: 700).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Random seeds (space-separated).")
    parser.add_argument("--variant", type=str, default=None,
                        choices=[c["name"] for c in ABLATIONS] + [None],
                        help="Run a single ablation by name; default: all.")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to a JSONL telemetry file to feed the loop (optional).")
    args = parser.parse_args()

    outdir = Path("data/outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_all(
        variant=args.variant,
        epochs=args.epochs,
        seeds=args.seeds,
        jsonl_path=Path(args.jsonl) if args.jsonl else None
    )
    (outdir / "epochs_log.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "epochs_log.csv", index=False)
    save_summary_and_csv(df, out_csv=outdir / "table_main.csv")
    print(f"Logs saved to {outdir / 'epochs_log.csv'}")


if __name__ == "__main__":
    cli()
