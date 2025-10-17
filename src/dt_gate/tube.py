import math
def tube_radius_from_nis(nis_p95, gain=1.0):
    base=0.15
    return base * (1.0 + 0.5*min(3.0, math.sqrt(max(nis_p95,1e-6)))) / max(1e-6,gain)
