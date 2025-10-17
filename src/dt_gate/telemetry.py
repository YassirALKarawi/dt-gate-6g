from dataclasses import dataclass
import numpy as np
from .queueing import kingman_wait

@dataclass
class Cls:
    lmbd: float; mu: float; c2a: float; c2s: float

def init_classes():
    return {"URLLC":Cls(0.18,0.22,1.2,1.1),
            "eMBB": Cls(0.12,0.16,1.4,1.3),
            "mMTC": Cls(0.06,0.09,1.1,1.1)}

def attack(t, period=160):
    act = 1.0 if 40 <= (t%period) < 80 else 0.0
    return {"active":act, "lambda_boost":0.6*act, "mu_drop":0.25*act, "jitter":0.5*act}

def plant_epoch(state, control_gain, atk, energy_base):
    obs={}
    energy = energy_base*(1-0.15*control_gain); energy=max(0.6,energy)
    labels_attack = 1 if atk["active"]>0.5 else 0
    for k,p in state.items():
        lmbd=max(1e-4, p.lmbd*(1+atk["lambda_boost"]))
        mu   =max(1e-4, p.mu*(1-atk["mu_drop"]))*(1+0.20*control_gain)
        c2s_e=max(0.8, p.c2s*(1-0.20*control_gain))
        w=kingman_wait(lmbd,mu,p.c2a,c2s_e)
        muD=w + (1.0/max(mu,1e-9)) + 0.25 + 0.15
        sig=0.35*muD*(1+0.30*atk["jitter"])
        obs[k]=(muD, sig, lmbd, mu)
    util = np.mean([min(0.999, o[2]/max(o[3],1e-9)) for o in obs.values()])
    return obs, energy, util, labels_attack
