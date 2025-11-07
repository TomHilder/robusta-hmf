import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm

rng = np.random.default_rng(0)


def quantile_weights(p, Q=1):
    k = np.abs(stats.Normal().icdf(p))
    return Q**2 / (Q**2 + k**2)


Q_vals = np.linspace(1, 2, 2)
p_vals = np.linspace(1e-2, 1 - 1e-2, 200)

Q, p = np.meshgrid(Q_vals, p_vals)
q_W = quantile_weights(p, Q)

plt.figure(figsize=(8, 6))
contour = plt.contourf(Q, p, q_W, levels=50, cmap="viridis")
plt.colorbar(contour, label="Quantile $q_W$")
plt.xlabel("Robust Parameter $Q$")
plt.ylabel("Quantile probability $p$")
plt.title("Quantile Weights as a Function of $Q$ and $p$")
# plt.yscale('logit')
# plt.xscale('log')
plt.ylim(1e-2, 1 - 1e-2)
plt.show()


# Slice plot for fixed Q values
plt.figure(figsize=(8, 6))
for Q in [1]:
    q_W_slice = quantile_weights(p_vals, Q)
    plt.plot(p_vals, q_W_slice, label=f"$Q={Q}$")

plt.xlabel("$p$")
plt.ylabel("$p$th quantile of weights distribution")
plt.show()

draws = rng.normal(size=(100000,), loc=0, scale=1)
Q = 1
weights = Q**2 / (Q**2 + draws**2)

plt.figure(figsize=(8, 6))
plt.hist(weights, bins=50, density=True, alpha=0.7)
plt.xlabel("Weight")
plt.ylabel("Density")
plt.title(f"Histogram of Weights for Q={Q}")
plt.show()
