import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from matplotlib.axes import Axes

def one_side_test_z_score(n: int, alpha: float, mu: float, x_n: float, std: float) -> None:
    z_score = np.sqrt(n) * (x_n - mu) / std
    p_value = 1 - norm.cdf(z_score)

    print(f"z-score: {z_score}")
    print(f"p-value: {p_value}")
    if p_value < alpha:
        print("Null hypothesis is rejected")
    else:
        print("Null hypothesis is not rejected")

def one_side_test_crit_zone(ax: Axes, n: int, alpha: float, mu: float, x_n: float, std: float) -> None:
    z_score = np.sqrt(n) * (x_n - mu) / std
    crit = norm.ppf(1 - alpha)

    print(f"x_n: {x_n}")
    print(f"crit: {crit}")
    if z_score > crit:
        print("Null hypothesis is rejected")
    else:
        print("Null hypothesis is not rejected")

    std_normal_range = 4
    x = np.linspace(-std_normal_range, std_normal_range, 100)
    y = norm.pdf(x)
    ax.plot(x, y)
    ax.axvline(z_score, color="green", label=f"Z-score - {z_score} (x_n: {x_n})")
    ax.fill_between(x, 0, y, where=(x >= crit), alpha=0.5, color='red')
    ax.legend()

def two_side_test_z_score(n: int, alpha: float, mu: float, x_n: float, std: float) -> None:
    z_score = np.sqrt(n) * (x_n - mu) / std
    p_value = 2 * (1 - norm.cdf(z_score))

    print(f"z-score: {z_score}")
    print(f"p-value: {p_value}")
    if p_value < alpha:
        print("Null hypothesis is rejected")
    else:
        print("Null hypothesis is not rejected")

def two_side_test_crit_zone(ax: Axes, n: int, alpha: float, mu: float, x_n: float, std: float) -> None:
    z_score = np.sqrt(n) * (x_n - mu) / std
    crit = norm.ppf(1 - alpha / 2)

    print(f"z_score: {z_score}")
    print(f"crit: {crit}")
    if abs(z_score) > crit:
        print("Null hypothesis is rejected")
    else:
        print("Null hypothesis is not rejected")

    std_normal_range = 4
    x = np.linspace(-std_normal_range, std_normal_range, 100)
    y = norm.pdf(x)
    ax.plot(x, y)
    ax.axvline(z_score, color="green", label=f"Z-score - {z_score} (x_n: {x_n})")
    ax.fill_between(x, 0, y, where=(x >= crit) | (x <= -crit), alpha=0.5, color='red')
    ax.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
one_side_test_z_score(n=10, alpha=0.05, mu=10, x_n=12, std=5)
one_side_test_crit_zone(ax=ax1, n=10, alpha=0.15, mu=10, x_n=12, std=5)
two_side_test_z_score(n=10, alpha=0.05, mu=10, x_n=12, std=5)
two_side_test_crit_zone(ax=ax2, n=10, alpha=0.15, mu=10, x_n=12, std=5)
plt.show()