import math


def truncation_coef(n: float, m: float, gamma: float = 1) -> float:
    beta = float(m)/float(n)
    return (2*(beta + 1) + ((8*beta)/(beta + 1) + np.sqrt(beta**2 + 14*beta + 1)))*math.sqrt(n)*gamma