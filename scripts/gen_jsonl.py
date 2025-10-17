#!/usr/bin/env python3
import argparse, json, os, random, numpy as np
from pathlib import Path

# reuse your plant model so the JSON matches the sim
from src.dt_gate.telemetry import init_classes, attack, plant_epoch
from src.dt_gate.config import ENERGY_BASE

def main():
    ap = argparse.ArgumentParser(description="Generate JSONL telemetry")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gain", type=float, default=0.0, help="fixed control gain used when generating")
    ap.add_argument("--outfile", type=str, default="data/telemetry/sim.jsonl")
    args = ap.parse_args()

    np.random.seed(args.seed); random.seed(args.seed)
    state = init_classes()

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for t in range(args.epochs):
            atk = attack(t)
            obs, energy, util, _ = plant_epoch(
                state=state,
                control_gain=args.gain,
                atk=atk,
                energy_base=ENERGY_BASE
            )
            rec = {
                "t": t,
                "classes": {
                    k: {"muD": float(muD), "sigD": float(sig), "lambda": float(lmbd), "mu": float(mu)}
                    for k, (muD, sig, lmbd, mu) in obs.items()
                },
                "util": float(util),
                "attack": {k: float(v) for k, v in atk.items()},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {args.epochs} records → {out}")

if __name__ == "__main__":
    main()

