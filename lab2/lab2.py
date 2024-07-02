import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, bernoulli, binom, norm, ecdf

# Построение эмпирической CDF по выборке data с точками bins
def emp_cdf(data, bins):
    s_data = sorted(data)
    data_len = len(data)

    bins_len = len(bins)
    bins_idx = 0

    emp_cdf_list = [0] * bins_len

    for elem in s_data:
        while bins_idx != (bins_len - 1) and elem > bins[bins_idx]:
            bins_idx += 1
            emp_cdf_list[bins_idx] = emp_cdf_list[bins_idx - 1]
        emp_cdf_list[bins_idx] += 1

    while bins_idx != (bins_len - 1):
        bins_idx += 1
        emp_cdf_list[bins_idx] = emp_cdf_list[bins_idx - 1]

    func_divide = np.vectorize(lambda x: x / data_len)

    return func_divide(emp_cdf_list)          

# Построение эмпирического доверительного интервала по выборке data 
# с точками bins и с вероятностью покрытия исходной CDF в каждой точке a.
# Доверительный интервал считается по неравентсву   
# в результате CDF покрывается в каждой точке вместе с вероятностью a
def emp_conf_int(data, bins, a):
    emp_cdf_list = emp_cdf(data, bins)
    b = 1 - a
    n = len(bins)
    l, r = [], []

    for elem in emp_cdf_list:
        eps = np.sqrt(np.log(2 / b) / (2 * n))
        l.append(max(elem - eps, 0))
        r.append(min(elem + eps, 1))

    return (l, r)

# Добавялем к графику...
# ...самописную эмпирическую CDF
def add_my_emp_cdf(ax, data, bins):
    ax.plot(bins, emp_cdf(data, bins), color="blue", label="My emp. CDF")
# ...либовскую эмпирическую CDF
def add_py_emp_cdf(ax, x, dist):
    ax.plot(x, dist.cdf(x), color="red", label="Py emp. CDF")
# ...самописный доверительный интервал
def add_my_conf_int(ax, data, a, bins):
    l, r = emp_conf_int(data, bins, a)
    ax.plot(bins, l, color="green", linestyle="dashed", label="My conf. int")
    ax.plot(bins, r, color="green", linestyle="dashed")
# ...либовский доверительный интервал, который покрывает каждую
# точку CDF в отдельности с шансом a
def add_py_conf_int(ax, data, a):
    l, r = ecdf(data).cdf.confidence_interval(a)
    ax.plot(l.quantiles, l.probabilities, color="black", linestyle="dotted", label="Py conf. int")
    ax.plot(r.quantiles, r.probabilities, color="black", linestyle="dotted")

# Прикрепить к графику данные в задании графики:
# Равномерное непрерывное 
def show_uniform(ax, a, b, data_count, x_count, conf_int_a):
    n_a = a
    n_b = b - a
    u = uniform(n_a, n_b)
    data = u.rvs(data_count)

    x = np.linspace(a, b, x_count)

    ax.set_title(f"Uniform (data count: {data_count})")
    add_my_emp_cdf(ax, data, x)
    add_py_emp_cdf(ax, x, u)
    add_my_conf_int(ax, data, conf_int_a, x)
    add_py_conf_int(ax, data, conf_int_a)

# Бернулли
def show_bernoulli(ax, a, data_count, conf_int_a):
    b = bernoulli(a)
    data = b.rvs(data_count)

    x = np.array([0, 1])

    ax.set_title(f"Bernoulli (data count: {data_count})")
    add_my_emp_cdf(ax, data, x)
    add_py_emp_cdf(ax, x, b)
    add_my_conf_int(ax, data, conf_int_a, x)
    add_py_conf_int(ax, data, conf_int_a)

# Биномиальное
def show_binom(ax, n, p, data_count, conf_int_a):
    b = binom(n, p)
    data = b.rvs(data_count)

    x = np.arange(n + 1)

    ax.set_title(f"Binomial (data count: {data_count})")
    add_my_emp_cdf(ax, data, x)
    add_py_emp_cdf(ax, x, b)
    add_my_conf_int(ax, data, conf_int_a, x)
    add_py_conf_int(ax, data, conf_int_a)

# Нормальное
def show_norm(ax, mean, es, data_count, x_count, conf_int_a):
    n = norm(mean, es)
    data = n.rvs(data_count)

    x = np.linspace(mean - 5 * es, mean + 5 * es, x_count)

    ax.set_title(f"Normal (data count: {data_count})")
    add_my_emp_cdf(ax, data, x)
    add_py_emp_cdf(ax, x, n)
    add_my_conf_int(ax, data, conf_int_a, x)
    add_py_conf_int(ax, data, conf_int_a)

fig, ax = plt.subplots(1, 1)

# show_uniform(ax=ax,
#              a=0,
#              b=1,
#              data_count=100,
#              x_count=100,
#              conf_int_a=0.95)

# show_bernoulli(ax=ax,
#                a=0.8,
#                data_count=100,
#                conf_int_a=0.95)

# show_binom(ax=ax,
#            n=10,
#            p=0.3,
#            data_count=100,
#            conf_int_a=0.95)

# show_norm(ax=ax, 
#           mean=10,
#           es=2,
#           data_count=100,
#           x_count=100,
#           conf_int_a=0.95)

# show_uniform(ax=ax,
#              a=0,
#              b=1,
#              data_count=1000,
#              x_count=1000,
#              conf_int_a=0.95)

# show_bernoulli(ax=ax,
#                a=0.8,
#                data_count=1000,
#                conf_int_a=0.95)

# show_binom(ax=ax,
#            n=10,
#            p=0.3,
#            data_count=1000,
#            conf_int_a=0.95)

# show_norm(ax=ax, 
#           mean=10,
#           es=2,
#           data_count=1000,
#           x_count=1000,
#           conf_int_a=0.95)

ax.legend(loc='best', frameon=False)
fig.show()
fig.waitforbuttonpress()