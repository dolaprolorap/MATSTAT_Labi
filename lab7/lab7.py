import numpy as np
import scipy.stats as st

dc = 4

def KS_one_sample(data: list, dist: st.rv_continuous, alpha: float = 0.05):
    data_ecdf = st.ecdf(data)
    data_sorted = np.sort(data)
    d = 0
    for idx, sample in enumerate(data_sorted):
        d = max(d, abs(data_ecdf.cdf.probabilities[idx] - dist.cdf(sample)))
    en = np.sqrt(len(data))
    ks_test = en * d
    ks_crit = st.kstwobign.ppf(1 - alpha)
    p_value = 1 - st.kstwobign.cdf(ks_test)

    print(f"=== KS Test ===")
    print(f"KS test, by crit zone: {"Different" if ks_test > ks_crit else "Not different"}")
    print(f"ks_test: {round(ks_test, dc)}, ks_crit: {round(ks_crit, dc)}")
    print(f"KS test, by p_value: {"Different" if p_value < alpha else "Not different"}")
    print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")

def KS_two_samples(data1: list, data2: list, alpha: float = 0.05):
    data1_ecdf = st.ecdf(data1)
    data2_ecdf = st.ecdf(data2)
    x = np.linspace(-10, 10, 1000)
    d = 0
    for x_s in x:
        d = max(d, abs(data1_ecdf.cdf.evaluate(x_s) - data2_ecdf.cdf.evaluate(x_s)))
    ks_crit = st.kstwobign.ppf(1 - alpha)
    en = np.sqrt(len(data1) * len(data2) / (len(data1) + len(data2)))
    ks_test = en * d
    p_value = 1 - st.kstwobign.cdf(ks_test)
    print(f"=== KS Test (2 samples) ===")
    print(f"KS test (2 samples), by crit zone: {"Different" if ks_test > ks_crit else "Not different"}")
    print(f"ks_test: {round(ks_test, dc)}, ks_crit: {round(ks_crit, dc)}")
    print(f"KS test (2 samples), by p_value: {"Different" if p_value < alpha else "Not different"}")
    print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")

if __name__ == "__main__":
    norm1 = st.norm(loc=0, scale=1)
    norm2 = st.norm(loc=1, scale=2)

    data_u1 = np.random.uniform(0, 1, 1000)
    data_u2 = np.random.uniform(0.5, 1.3, 1000)
    data_n1 = norm1.rvs(1000)
    data_n2 = norm2.rvs(1000)
    # KS_one_sample(data_n2, norm1)
    # print(st.ks_1samp(data_n2, norm1.cdf).pvalue)
    # print(st.shapiro(data_u1).pvalue)
    KS_two_samples(norm1.rvs(100), norm1.rvs(1000))