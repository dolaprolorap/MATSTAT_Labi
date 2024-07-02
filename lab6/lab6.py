import numpy as np
import scipy.stats as st

# Decimals count in round method
dc = 4

def large_sample_test(data1: list, data2: list, type: str, allow_small_samples: bool = False, alpha: float = 0.05) -> None:
    if not(allow_small_samples):
        if len(data1) < 30:
            raise ValueError("data1 length < 30")
        if len(data2) < 30:
            raise ValueError("data2 length < 30")
        
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")

    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    n1 = len(data1)
    n2 = len(data2)

    z = (mean1 - mean2) / np.sqrt(var1 / n1 + var2 / n2)
    if type == two_tailed_name:
        z_crit = st.norm.ppf(1 - alpha / 2)
        p_value = 2 * (1 - st.norm.cdf(abs(z)))
        print(f"=== Two tailed large sample test ===")
        print(f"Large sample test, by z-criterion: {"Rejected" if abs(z) > z_crit else "Not rejected"}")
        print(f"z: {round(z, dc)}, z_crit: +-{round(z_crit, dc)}")
        print(f"Large sample test, by p-value: {"Rejected" if p_value < alpha else "Not rejected"}")
        print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")
    elif type == left_tailed_name:
        z_crit = st.norm.ppf(alpha)
        p_value = st.norm.cdf(z)
        print(f"=== Left tailed large sample test ===")
        print(f"Large sample test, by z-criterion: {"Rejected" if z < z_crit else "Not rejected"}")
        print(f"z: {round(z, dc)}, z_crit: {round(z_crit, dc)}")
        print(f"Large sample test, by p-value: {"Rejected" if p_value < alpha else "Not rejected"}")
        print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")
    else:
        z_crit = st.norm.ppf(1 - alpha)
        p_value = 1 - st.norm.cdf(z)
        print(f"=== Right tailed large sample test ===")
        print(f"Large sample test, by z-criterion: {"Rejected" if z > z_crit else "Not rejected"}")
        print(f"z: {round(z, dc)}, z_crit: {round(z_crit, dc)}")
        print(f"Large sample test, by p-value: {"Rejected" if p_value < alpha else "Not rejected"}")
        print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")

def f_test(data1: list, data2: list, type: str, alpha: float = 0.05) -> bool:
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")
    
    if len(data1) >= len(data2):
        big_data = data1
        small_data = data2
    else:
        big_data = data2
        small_data = data1

    mean_big = np.mean(big_data)
    mean_small = np.mean(small_data)
    var_big = np.var(big_data, ddof=1)
    var_small = np.var(small_data, ddof=1)
    df_big = len(big_data) - 1
    df_small = len(small_data) - 1

    f_test = var_big / var_small
    f = st.f(dfn=df_big, dfd=df_small)

    if type == "two-tailed":
        left_f_crit = f.ppf(alpha / 2)
        right_f_crit = f.ppf(1 - alpha / 2)
        p_value = 2 * min(f.cdf(f_test), 1 - f.cdf(f_test))
        print(f"=== Two tailed F test ===")
        print(f"F test, by f-crit zone: {"Rejected" if f_test < left_f_crit or f_test > right_f_crit else "Not rejected"}")
        print(f"f_test: {round(f_test, dc)}, left_f_crit: {round(left_f_crit, dc)}, right_f_crit: {round(right_f_crit, dc)}")
        print(f"F test, by p-value: {"Rejected" if p_value < alpha else "Not rejected"}")
        print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")
    elif type == "left-tailed":
        f_crit = f.ppf(alpha)
        p_value = f.cdf(f_test)
        print(f"=== Left tailed F test ===")
        print(f"F test, by f-crit zone: {"Rejected" if f_test < f_crit else "Not rejected"}")
        print(f"f_test: {round(f_test, dc)}, f_crit: {round(f_crit, dc)}")
        print(f"F test, by p-value: {"Rejected" if p_value < alpha else "Not rejected"}")
        print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")
    else:
        f_crit = f.ppf(1 - alpha)
        p_value = 1 - f.cdf(f_test)
        print(f"=== Right tailed F test ===")
        print(f"F test, by f-crit zone: {"Rejected" if f_test > f_crit else "Not rejected"}")
        print(f"f_test: {round(f_test, dc)}, f_crit: {round(f_crit, dc)}")
        print(f"F test, by p-value: {"Rejected" if p_value < alpha else "Not rejected"}")
        print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")
    return p_value >= alpha

def t_test_my(data1: list, data2: list, type: str, alpha: float = 0.05):
    if not f_test(data1, data2, type, alpha):
        raise ValueError("Variances of data1 and data2 are not the same (by f_test)")
    
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    df1 = len(data1)
    df2 = len(data2)

    pooled_var = ((df1 - 1) * var1 + (df2 - 1) * var2) / (df1 + df2 - 2)

    t_test = (mean1 - mean2) / np.sqrt(pooled_var * (1 / df1 + 1 / df2))

def t_test(data1: list, data2: list, type: str, alpha: float = 0.05):
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")
    
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    df1 = len(data1)
    df2 = len(data2)
    if f_test(data1, data2, type, alpha):
        df = df1 + df2 - 2
    else:
        n = (var1 / df1 + var2 / df2) ** 2
        d = ((var1 / df1) ** 2) / (df1 - 1) + ((var2 / df2) ** 2) / (df2 - 1)
        df = n / d

    t_statistic, p_value = st.ttest_ind(data1, data2, equal_var=False, alternative={
        two_tailed_name: "two-sided",
        left_tailed_name: "less",
        right_tailed_name: "greater"
    }[type])
    t = st.t(df=df)
    
    if type == two_tailed_name:
        left_crit = t.ppf(alpha / 2)
        right_crit = t.ppf(1 - alpha / 2)
        print(f"=== Two tailed T test ===")
        print(f"T test, by t-crit zone: {"Rejected" if t_statistic > right_crit or t_statistic < left_crit else "Not rejected"}")
        print(f"t_statistic: {round(t_statistic, dc)}, l_crit: {round(left_crit, dc)}, r_crit: {round(right_crit, dc)}")
    elif type == left_tailed_name:
        crit = t.ppf(alpha)
        print(f"=== Left tailed T test ===")
        print(f"T test, by t-crit zone: {"Rejected" if t_statistic < crit else "Not rejected"}")
        print(f"t_statistic: {round(t_statistic, dc)}, crit: {round(crit, dc)}")
    else: 
        crit = t.ppf(1 - alpha)
        print(f"=== Right tailed T test ===")
        print(f"T test, by t-crit zone: {"Rejected" if t_statistic > crit else "Not rejected"}")
        print(f"t_statistic: {round(t_statistic, dc)}, crit: {round(crit, dc)}")
    print(f"T test, by p_value: {"Rejected" if p_value < alpha else "Not rejected"}")
    print(f"p_value: {round(p_value, dc)}, alpha: {round(alpha, dc)}")

if __name__ == "__main__":
    norm1 = st.norm(10, 0.5)
    norm2 = st.norm(10.04, 0.489)
    data1 = norm1.rvs(1000)
    data2 = norm2.rvs(1000)
    large_sample_test(data1, data2, "two-tailed")
    f_test(data1, data2, "right-tailed")
    t_test(data1, data2, "two-tailed")

    