import argparse, random, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from .config import *
from .queueing import beta_cantelli, bootstrap_ci
from .ekf import EKFScalar
from .tube import tube_radius_from_nis
from .risk import RiskMLP, isotonic_calibrate, smooth_samples
from .optimizer import raso_solve
from .telemetry import init_classes, attack, plant_epoch

ABLATIONS = [
    {"name":"Proposed","tube":True,"cvar":True,"smooth":True,"trust_gate":True},
    {"name":"-- Tube MPC","tube":False,"cvar":True,"smooth":True,"trust_gate":True},
    {"name":"-- CVaR term","tube":True,"cvar":False,"smooth":True,"trust_gate":True},
    {"name":"-- Smoothing","tube":True,"cvar":True,"smooth":False,"trust_gate":True},
    {"name":"-- Trust gate τ_min","tube":True,"cvar":True,"smooth":True,"trust_gate":False},
]

def build_feat(obs, util, atk):
    feats=[]
    for k,(muD, sigD, lmbd, mu) in obs.items():
        feats += [muD, sigD, lmbd, mu]
    feats += [util, atk["active"], atk["lambda_boost"], atk["mu_drop"], atk["jitter"]]
    return np.array(feats[:10], dtype=float)

def run_once(seed, cfg):
    np.random.seed(seed); random.seed(seed)
    state=init_classes(); beta={k:beta_cantelli(EPS[k]) for k in CLASSES}
    ekf=EKFScalar(mu0=10.0,P0=4.0,Q=0.2,R=1.0)
    mlp=RiskMLP(din=10, dh=16, seed=seed)
    last_safe_gain=0.0
    logs=[]; raw_scores=[]; labels=[]
    nis_hist=[]
    # warm-up for isotonic
    for t in range(150):
        atk=attack(t); obs,energy,util,label=plant_epoch(state,last_safe_gain,atk, ENERGY_BASE)
        feat=build_feat(obs,util,atk); s=mlp.forward(feat); raw_scores.append(s); labels.append(label)
    calib=isotonic_calibrate(raw_scores, labels)

    for t in range(EPOCHS):
        atk=attack(t); obs,energy,util,label=plant_epoch(state,last_safe_gain,atk, ENERGY_BASE)
        mu_meas = obs["URLLC"][0]
        tau, NIS, muP, PP = ekf.step(mu_meas)
        nis_hist.append(NIS)

        feat=build_feat(obs,util,atk); s_raw=mlp.forward(feat)
        s_cal=calib.predict([np.clip(s_raw,0,1)])[0]
        risks = smooth_samples(s_cal, M_SMOOTH if cfg["smooth"] else 1, SIGMA_S if cfg["smooth"] else 0.0)

        gate_ok = ((tau>=TAU_MIN) and (np.mean(risks)<=R_MAX)) if cfg["trust_gate"] else True

        nis_p95 = float(np.quantile(nis_hist[-100:] if len(nis_hist)>=5 else nis_hist, 0.95))
        tube_m = 0.0 if not cfg["tube"] else tube_radius_from_nis(nis_p95, gain=max(0.3,last_safe_gain+0.3))

        best = raso_solve(
            obs, beta,
            params={"tube_margin":tube_m,"NIS":NIS,"D_BUDGET_MS":D_BUDGET_MS},
            risks=risks,
            cfg=cfg,
            weights=W, energy_base=ENERGY_BASE, alpha=ALPHA, eta=ETA, gamma=GAMMA, zeta=ZETA
        )
        if best["feas"] and gate_ok:
            gain=best["gain"]; last_safe_gain=gain; accepted=1
        else:
            gain=last_safe_gain; accepted=0

        Ls=[]; viol={}
        for k,(muD,sig,_,_) in obs.items():
            mu_e = muD*(1-0.25*gain); sig_e=sig*(1-0.20*gain)
            viol[k] = 1 if (mu_e + beta[k]*sig_e) > D_BUDGET_MS[k] else 0
            Ls.append(mu_e + 2.326*sig_e)
        L99=float(max(Ls))
        energy_e = ENERGY_BASE*(1-0.15*gain); energy_e=max(0.6,energy_e)

        logs.append({"t":t,"gain":gain,"accepted":accepted,"tau":tau,"risk":float(np.mean(risks)),"L99":L99,
                     "energy":energy_e,"viol_url":viol["URLLC"],"label_atk":label,"tube_m":tube_m})
    return pd.DataFrame(logs), calib

def run_all(variant=None, epochs=EPOCHS, seeds=SEEDS):
    global EPOCHS
    EPOCHS = epochs
    cfgs = [c for c in ABLATIONS if (variant is None or c["name"]==variant)]
    all_frames=[]
    for cfg in cfgs:
        for s in seeds:
            df,_=run_once(s, cfg)
            df["seed"]=s; df["variant"]=cfg["name"]
            all_frames.append(df)
    return pd.concat(all_frames, ignore_index=True)

def save_summary_and_csv(df, out_csv="data/outputs/table_main.csv"):
    table=[]
    for v in df["variant"].unique():
        sub=df[df["variant"]==v]
        L=sub["L99"].values; E=sub["energy"].values
        auc = float('nan')
        try: auc = roc_auc_score(sub["label_atk"].values, 1.0-sub["risk"].values)
        except: pass
        Lm, Llo, Lhi=bootstrap_ci(L); Em, Elo, Ehi=bootstrap_ci(E)
        table.append({
            "Method":v,
            "L99_mean":Lm,"L99_lo":Llo,"L99_hi":Lhi,
            "SLA_URLLC_%":100*sub["viol_url"].mean(),
            "Energy_J/MB":Em,"E_lo":Elo,"E_hi":Ehi,
            "AUC":auc,"GateAccept_%":100*sub["accepted"].mean()
        })
    out=pd.DataFrame(table)
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {out_csv}")

def cli():
    p=argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=700)
    p.add_argument("--seeds", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--variant", type=str, default=None)
    args=p.parse_args()
    df = run_all(variant=args.variant, epochs=args.epochs, seeds=args.seeds)
    # ensure output dir exists
    import os
    os.makedirs("data/outputs", exist_ok=True)
    df.to_csv("data/outputs/epochs_log.csv", index=False)
    save_summary_and_csv(df)
    print("Logs saved to data/outputs/epochs_log.csv")

if __name__ == '__main__':
    cli()
