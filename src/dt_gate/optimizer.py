import numpy as np
from .risk import cvar_emp

def raso_solve(obs, beta, params, risks, cfg, weights, energy_base, alpha, eta, gamma, zeta):
    tube_margin = 0.0 if not cfg.get("tube", True) else params.get("tube_margin",0.0)
    risks = risks if cfg.get("smooth", True) else risks[:1]
    best={"gain":0.0,"obj":float("inf"),"feas":False}
    for gain in np.linspace(0,1,21):
        feas=True; cost_delay=0.0
        for k,(muD, sigD, lmbd, mu) in obs.items():
            mu_e = muD*(1-0.25*gain); sig_e = sigD*(1-0.20*gain)
            lhs  = mu_e + beta[k]*sig_e + tube_margin
            if lhs > params["D_BUDGET_MS"][k]: feas=False
            cost_delay += weights[k]*mu_e
        energy = energy_base*(1-0.15*gain); energy=max(0.6,energy)
        cvar = 0.0 if not cfg.get("cvar", True) else cvar_emp(risks, alpha)
        obj = cost_delay + eta*energy + gamma*cvar + zeta*params["NIS"]
        if feas and obj < best["obj"]: best={"gain":float(gain),"obj":float(obj),"feas":True}
    if not best["feas"]: best={"gain":0.0,"obj":float("inf"),"feas":False}
    return best
