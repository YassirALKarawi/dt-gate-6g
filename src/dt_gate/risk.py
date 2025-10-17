import numpy as np
from sklearn.isotonic import IsotonicRegression

class RiskMLP:
    def __init__(self, din=10, dh=16, seed=0):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0,0.6,(din,dh)); self.b1=np.zeros(dh)
        self.W2=rng.normal(0,0.4,(dh,1)); self.b2=np.zeros(1)
        self.temp=1.0
    def _relu(self,x): return np.maximum(0,x)
    def forward(self, x):
        h=self._relu(x@self.W1 + self.b1); z=h@self.W2 + self.b2
        s=1.0/(1.0+np.exp(-z/self.temp))
        return float(np.clip(s,0,1))

def isotonic_calibrate(raw_scores, labels):
    xs=np.asarray(raw_scores); ys=np.asarray(labels)
    xs=np.clip(xs,0,1)
    ir=IsotonicRegression(y_min=0.0,y_max=1.0,out_of_bounds="clip")
    ir.fit(xs, ys)
    return ir

def smooth_samples(score, M, sigma):
    return np.clip(score + np.random.normal(0,sigma,size=M),0,1)

def cvar_emp(samples, alpha):
    if len(samples)==0: return 0.0
    z=np.sort(np.asarray(samples)); nu=z[int(np.floor(alpha*(len(z)-1)))]
    tail=np.clip(z-nu,0,None)
    return float(nu + tail.mean()/(1-alpha))
