import math
class EKFScalar:
    def __init__(self, mu0=10.0, P0=4.0, Q=0.1, R=1.0):
        self.mu=mu0; self.P=P0; self.Q=Q; self.R=R
    def step(self, y):
        mu_p=self.mu; P_p=self.P + self.Q
        innov=y - mu_p; S=P_p + self.R
        K=P_p / S
        self.mu = mu_p + K*innov
        self.P  = (1-K)*P_p
        NIS = (innov**2)/S
        tau = math.exp(-0.5*NIS)
        return tau, NIS, self.mu, self.P
