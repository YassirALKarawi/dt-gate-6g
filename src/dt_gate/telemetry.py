from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from .queueing import kingman_wait

@dataclass
class Cls:
    lmbd: float  # arrival rate
    mu: float    # service rate
    c2a: float   # SCV of arrivals
    c2s: float   # SCV of service

def init_classes() -> Dict[str, Cls]:
    return {
        "URLLC": Cls(0.18, 0.22, 1.2, 1.1),
        "eMBB":  Cls(0.12, 0.16, 1.4, 1.3),
        "mMTC":  Cls(0.06, 0.09, 1.1, 1.1),
    }

def attack(t: int, period: int = 160) -> Dict[str, float]:
    """Periodic attack (active for 40 <= t%period < 80)."""
    act = 1.0 if 40 <= (t % period) < 80 else 0.0
    return {
        "active": act,
        "lambda_boost": 0.60 * act,  # +60% arrival rate
        "mu_drop": 0.25 * act,       # -25% service rate
        "jitter": 0.50 * act,        # more delay variance
    }

def plant_epoch(
    state: Dict[str, Cls],
    control_gain: float,
    atk: Dict[str, float],
    energy_base: float
) -> Tuple[Dict[str, Tuple[float, float, float, float]], float, float, int]:
    """
    Returns:
      obs[k] = (muD, sigD, lambda, mu)
      energy in [0.6, 1.0]
      util in [0, 0.999]
      label_attack in {0,1}
    """
    obs: Dict[str, Tuple[float, float, float, float]] = {}

    # execution energy (bounded)
    energy = max(0.6, min(1.0, energy_base * (1 - 0.15 * control_gain)))

    label_attack = 1 if atk.get("active", 0.0) > 0.5 else 0

    for k, p in state.items():
        lmbd = max(1e-6, p.lmbd * (1 + atk.get("lambda_boost", 0.0)))
        mu_raw = max(1e-6, p.mu * (1 - atk.get("mu_drop, ", 0.0)))  # typo intentionally? fix
        mu_raw = max(1e-6, p.mu * (1 - atk.get("mu_drop", 0.0)))    # <- correct line
        mu_eff = max(1e-6, mu_raw * (1 + 0.20 * control_gain))      # control increases μ

        # service SCV under control
        c2s_e = max(0.8, p.c2s * (1 - 0.20 * control_gain))

        # Kingman waiting time (make sure units match your delay budgets)
        w = kingman_wait(lmbd, mu_eff, p.c2a, c2s_e)

        # total delay proxy = wait + service + fixed overheads
        muD = w + (1.0 / mu_eff) + 0.25 + 0.15
        muD = float(max(1e-9, muD))

        # std dev proportional to delay with attack jitter
        sig = 0.35 * muD * (1 + 0.30 * atk.get("jitter", 0.0))
        sig = float(max(1e-9, sig))

        obs[k] = (muD, sig, lmbd, mu_eff)

    # average utilization across classes (clipped for stability)
    util_vals = [min(0.999, o[2] / max(o[3], 1e-9)) for o in obs.values()]
    util = float(np.mean(util_vals)) if util_vals else 0.0

    return obs, energy, util, label_attack
