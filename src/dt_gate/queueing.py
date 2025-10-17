import math
import numpy as np

def kingman_wait(lmbd, mu, c2a, c2s):
    rho = min(lmbd/max(mu,1e-9), 0.999999)
    sbar = 1.0/max(mu,1e-9)
    return ((c2a+c2s)/2.0) * (rho/(1.0-rho)) * sbar

def beta_cantelli(eps): 
    return math.sqrt((1-eps)/eps)

def bootstrap_ci(arr, B=1000, agg=np.mean):
    if len(arr)==0: return (float("nan"),)*3
    n=len(arr)
    idx=np.random.randint(0,n,(B,n))
    stats=[agg(arr[i]) for i in idx]
    m=float(np.mean(arr)); lo=float(np.quantile(stats,0.025)); hi=float(np.quantile(stats,0.975))
    return (m,lo,hi)
