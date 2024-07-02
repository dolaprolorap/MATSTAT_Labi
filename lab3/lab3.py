from scipy.stats import bootstrap, binom, norm
import numpy as np
import matplotlib.pyplot as plt 

bin_n = 10
bin_p = 0.6
norm_mean = 0
norm_std = 1
conf_level = 0.95

b = binom(bin_n, bin_p)
n = norm(norm_mean, norm_std)

bin_100 = b.rvs(100)
bin_1000 = b.rvs(1000)

norm_100 = n.rvs(100)
norm_1000 = n.rvs(1000)

# По методу моментов считаем p для бин. распределения
def mom_bin_p(data):
    return 1 - np.std(data) ** 2 / np.mean(data)

# Также считаем n
def mom_bin_n(data):
    return np.mean(data) / mom_bin_p(data)

def is_included_as_str(ci, val):
    if val >= ci[0] and val <= ci[1]:
        return "Yes"
    else:
        return "No"
    
def bin_p_bootstrap():
    bin_100_p_bootstrap_conf_int = bootstrap((bin_100, ), mom_bin_p, confidence_level=conf_level).confidence_interval
    bin_1000_p_bootstrap_conf_int = bootstrap((bin_1000, ), mom_bin_p, confidence_level=conf_level).confidence_interval

    print(f"===Bin p===")
    print(f"{conf_level}% CI for Bin 100, p: L:{bin_100_p_bootstrap_conf_int[0]} R:{bin_100_p_bootstrap_conf_int[1]}")
    print(f"True p of Bin 100: {bin_p}, is included in CI: {is_included_as_str(bin_100_p_bootstrap_conf_int, bin_p)}")
    print(f"{conf_level}% CI for Bin 1000, p: L:{bin_1000_p_bootstrap_conf_int[0]} R:{bin_1000_p_bootstrap_conf_int[1]}")
    print(f"True p of Bin 1000: {bin_p}, is included in CI: {is_included_as_str(bin_1000_p_bootstrap_conf_int, bin_p)}")

def bin_n_bootstrap():
    bin_100_n_bootstrap_conf_int = bootstrap((bin_100, ), mom_bin_n, confidence_level=conf_level).confidence_interval
    bin_1000_n_bootstrap_conf_int = bootstrap((bin_1000, ), mom_bin_n, confidence_level=conf_level).confidence_interval

    print(f"===Bin n===")
    print(f"{conf_level}% CI for Bin 100, n: L:{bin_100_n_bootstrap_conf_int[0]} R:{bin_100_n_bootstrap_conf_int[1]}")
    print(f"True n of Bin 100: {bin_n}, is included in CI: {is_included_as_str(bin_100_n_bootstrap_conf_int, bin_n)}")
    print(f"{conf_level}% CI for Bin 1000, n: L:{bin_1000_n_bootstrap_conf_int[0]} R:{bin_1000_n_bootstrap_conf_int[1]}")
    print(f"True n of Bin 1000: {bin_n}, is included in CI: {is_included_as_str(bin_1000_n_bootstrap_conf_int, bin_n)}")

def norm_mean_bootstrap():
    norm_100_mean_bootstrap_conf_int = bootstrap((norm_100, ), np.mean, confidence_level=conf_level).confidence_interval
    norm_1000_mean_bootstrap_conf_int = bootstrap((norm_1000, ), np.mean, confidence_level=conf_level).confidence_interval

    print(f"===Norm mean===")
    print(f"{conf_level}% CI for Norm 100, mean: L:{norm_100_mean_bootstrap_conf_int[0]} R:{norm_100_mean_bootstrap_conf_int[1]}")
    print(f"True mean of Norm 100: {norm_mean}, is included in CI: {is_included_as_str(norm_100_mean_bootstrap_conf_int, norm_mean)}")
    print(f"{conf_level}% CI for Norm 1000, mean: L:{norm_1000_mean_bootstrap_conf_int[0]} R:{norm_1000_mean_bootstrap_conf_int[1]}")
    print(f"True mean of Norm 1000: {norm_mean}, is included in CI: {is_included_as_str(norm_1000_mean_bootstrap_conf_int, norm_mean)}")

def norm_std_bootstrap():
    norm_100_std_bootstrap_conf_int = bootstrap((norm_100, ), np.std, confidence_level=conf_level).confidence_interval
    norm_1000_std_bootstrap_conf_int = bootstrap((norm_1000, ), np.std, confidence_level=conf_level).confidence_interval

    print(f"===Norm std===")
    print(f"{conf_level}% CI for Norm 100, std: L:{norm_100_std_bootstrap_conf_int[0]} R:{norm_100_std_bootstrap_conf_int[1]}")
    print(f"True std of Norm 100: {norm_std}, is included in CI: {is_included_as_str(norm_100_std_bootstrap_conf_int, norm_std)}")
    print(f"{conf_level}% CI for Norm 1000, std: L:{norm_1000_std_bootstrap_conf_int[0]} R:{norm_1000_std_bootstrap_conf_int[1]}")
    print(f"True std of Norm 1000: {norm_std}, is included in CI: {is_included_as_str(norm_1000_std_bootstrap_conf_int, norm_std)}")

def plot_norm():
    def plot_norm_n(ax, data):
        # ax.hist(data, 50)
        x = np.linspace(norm_mean - 5 * norm_std, norm_mean + 5 * norm_std, 100)
        ax.plot(x, n.pdf(x), label=f"Norm {len(data)} PDF")
        ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean (Norm {len(data)})")
        conf_int = bootstrap((data, ), np.mean, confidence_level=conf_level).confidence_interval
        ax.axvline(conf_int[0], color="green", linestyle="--",
                    label="CI (Bin 100)")
        ax.axvline(conf_int[1], color="green", linestyle="--")
        ax.axvline(norm_mean, color="purple", label=f"True mean")

    norm_100 = n.rvs(100)
    norm_1000 = n.rvs(1000)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_norm_n(ax1, norm_100)
    plot_norm_n(ax2, norm_1000)
    plt.legend()
    plt.show()

def mle():
    # Bin p, по формуле MLE
    bin_1000_p_mle = np.mean(bin_1000) / bin_n
    print(f"===MLE Bin 1000===")
    print(f"MLE Bin 1000 for p: {bin_1000_p_mle}")
    print(f"True p for Bin 1000: {bin_p}")

    # Norm mean, MLE
    norm_1000_mean_mle = np.mean(norm_1000)
    print(f"===MLE Norm 1000===")
    print(f"MLE Norm 1000 for mean: {norm_1000_mean_mle}")
    print(f"norm.fit[0] (loc) Norm 1000 for mean: {norm.fit(norm_1000)[0]}")
    print(f"True mean for Norm 1000: {norm_mean}")

    # Norm std, MLE
    norm_1000_std_mle = np.sqrt(np.mean((norm_1000 - norm_1000_mean_mle)**2))
    print(f"MLE Norm 1000 for std: {norm_1000_std_mle}")
    print(f"norm.fit[1] (scale) Norm 1000 for std: {norm.fit(norm_1000)[1]}")
    print(f"True std for Norm 1000: {norm_std}")
    

# bin_p_bootstrap()
# bin_n_bootstrap()
# norm_mean_bootstrap()
# norm_std_bootstrap()

# plot_norm()

mle()