import numpy as np
import scipy.stats as st
import pandas as pd

# Основные различия

# Метрика зависимости:
# Пирсон измеряет линейную зависимость.
# Спирмен измеряет монотонную зависимость.

# Тип данных:
# Пирсон подходит для количественных данных.
# Спирмен подходит для ранговых и количественных данных.

# Предположения:
# Пирсон требует нормальности и линейности.
# Спирмен не требует нормальности, может работать с нелинейными монотонными зависимостями.

# Чувствительность к выбросам:
# Пирсон чувствителен к выбросам.
# Спирмен менее чувствителен к выбросам, поскольку использует ранги вместо значений.

def check_norm(data: list, alpha: float = 0.05) -> bool:
    shapiro_p_value = st.shapiro(data).pvalue
    ks_p_value = st.ks_1samp(data, st.norm(loc=np.mean(data), scale=np.std(data)).cdf).pvalue
    return shapiro_p_value >= alpha and ks_p_value >= alpha

def correlation(data_x: list, data_y: list) -> float:
    if len(data_x) != len(data_y):
        raise ValueError("len(data_x) != len(data_y)")
    mean_x = np.mean(data_x)
    mean_y = np.mean(data_y)
    std_x = np.std(data_x, ddof=1)
    std_y = np.std(data_y, ddof=1)
    cov_xy = np.mean(data_x * data_y) - mean_x * mean_y
    return cov_xy / (std_x * std_y)

def spearman_pearson_t_value(r: float, n: float) -> float:
    return r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)

def pearson_r(data_x: list, data_y: list) -> float:
    return correlation(data_x, data_y)

def spearman_r(data_x: list, data_y: list) -> float:
    return correlation(st.rankdata(data_x), st.rankdata(data_y))

def pearson_t(data_x: list, data_y: list) -> float:
    if len(data_x) != len(data_y):
        raise ValueError("len(data_x) != len(data_y)")
    n = len(data_x)
    r = pearson_r(data_x, data_y)
    return spearman_pearson_t_value(r, n)

def spearman_t(data_x: list, data_y: list) -> float:
    if len(data_x) != len(data_y):
        raise ValueError("len(data_x) != len(data_y)")
    n = len(data_x)
    r = spearman_r(data_x, data_y)
    return spearman_pearson_t_value(r, n)

def crit_t(df: float, type: str, alpha: float = 0.05) -> tuple[float, float]:
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")
    
    t = st.t(df=df)
    
    if type == two_tailed_name:
        crit = t.ppf(1 - alpha / 2)
        return -float(crit), float(crit)
    elif type == left_tailed_name:
        crit = t.ppf(alpha)
        return float(crit), float(0)
    else:
        crit = t.ppf(1 - alpha)
        return float(crit), float(0)
    
def pearson(data_x: list, data_y: list, type: str = "two-tailed", alpha: float = 0.05):
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")

    if len(data_x) != len(data_y):
        raise ValueError("len(data_x) != len(data_y)")
    
    t_stat = pearson_t(data_x, data_y)
    df = len(data_x)
    t = st.t(df=df)
    p_value = 2 * (1 - t.cdf(t_stat))
    crit = crit_t(df, type, alpha)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "crit": crit,
        "r": float(pearson_r(data_x, data_y)),
        "rejected": bool(p_value < alpha)
    }

def spearman(data_x: list, data_y: list, type: str = "two-tailed", alpha: float = 0.05):
    two_tailed_name = "two-tailed"
    left_tailed_name = "left-tailed"
    right_tailed_name = "right-tailed"
    if type != two_tailed_name and type != left_tailed_name and type != right_tailed_name:
        raise ValueError(f"type must be \"{two_tailed_name}\", \"{left_tailed_name}\" or \"{right_tailed_name}\"")
    
    if len(data_x) != len(data_y):
        raise ValueError("len(data_x) != len(data_y)")
    
    t_stat = spearman_t(data_x, data_y)
    df = len(data_x)
    t = st.t(df=df)
    p_value = 2 * (1 - t.cdf(t_stat))
    crit = crit_t(df, type, alpha)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "crit": crit,
        "r": float(spearman_r(data_x, data_y)),
        "rejected": bool(p_value < alpha)
    }

if __name__ == "__main__":
    url_sociophobia = 'https://docs.google.com/spreadsheets/d/1YbW2NRoGYL0LwCuLaTfXT-PjoIAb2qnijczW8phlQEo/export?format=csv&gid=287922530'
    url_anxiety = 'https://docs.google.com/spreadsheets/d/1YbW2NRoGYL0LwCuLaTfXT-PjoIAb2qnijczW8phlQEo/export?format=csv&gid=855039775'
    url_fears = 'https://docs.google.com/spreadsheets/d/1YbW2NRoGYL0LwCuLaTfXT-PjoIAb2qnijczW8phlQEo/export?format=csv&gid=545882585'

    data_sociophobia = pd.read_csv(url_sociophobia)
    data_anxiety = pd.read_csv(url_anxiety)
    data_fears = pd.read_csv(url_fears)
    a = data_sociophobia["Общий"]
    b = data_anxiety["Общая"]

    fear_spiders = data_fears['Пауки'].tolist()[:89]
    fear_boss = data_fears['Начальство'].tolist()[:89]
    fear_future = data_fears['Будущее'].tolist()[:89]
    fear_responsibility = data_fears['Ответственность'].tolist()[:89]

    # print("=== Python method ===")
    # print(st.pearsonr(a, b))
    # print("=== My method ===")
    # print(pearson(a, b))

    # print("=== Python method ===")
    # print(st.spearmanr(fear_spiders, fear_boss))
    # print("=== My method ===")
    # print(spearman(fear_spiders, fear_boss))

    # print("=== Python method ===")
    # print(st.spearmanr(fear_future, fear_responsibility))
    # print("=== My method ===")
    # print(spearman(fear_future, fear_responsibility))

    print("=== Python method ===")
    print(st.pearsonr(fear_future, fear_responsibility))
    print("=== My method ===")
    print(pearson(np.array(fear_future), np.array(fear_responsibility)))