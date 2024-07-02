from scipy.stats import uniform, bernoulli, binom, norm, rv_discrete, rv_continuous
from numpy import average, var, median
import matplotlib.pyplot as plt

# Кол-во точек после запятой при округлении
ndigits = 3

# Класс-обвертка для распределений, включает навзвнаие распределения и само распределение
class NamedDist:
    def __init__(self, name : str, dist : rv_continuous | rv_discrete):
        self.name = name
        self.dist = dist

    # Вычисление среднего значения по выборке из count элементов
    def average(self, count : int) -> float:
        return average([round(k, ndigits) for k in self.dist.rvs(count)])
    
    # Вычисление дисперсии значения по выборке из count элементов
    def var(self, count : int) -> float:
        return var([round(k, ndigits) for k in self.dist.rvs(count)])
    
    # Вычисление стандартной ошибки по выборке из count элементов
    def se(self, count : int) -> float:
        return self.var(count)**0.5
    
    # Вычисление медианы по выборке из count элементов
    def median(self, count : int) -> float:
        return median([round(k, ndigits) for k in self.dist.rvs(count)])
    
    # Получение count элементов выборки
    def rvs(self, count : int) -> float:
        return self.dist.rvs(count)

cnts = [100, 1000]
dists = [
    NamedDist("Uniform", uniform(0, 1)), 
    NamedDist("Bernoulli", bernoulli(0.5)),
    NamedDist("Binom", binom(10, 0.5)),
    NamedDist("Norm", norm(0, 1))
]

for dist in dists:
    for cnt in cnts:
        print(f"=== {dist.name}({cnt}) ===")
        print(f"Average of {dist.name}({cnt})\t\t: {dist.average(cnt)}")
        print(f"Var of {dist.name}({cnt})\t\t: {dist.var(cnt)}")
        print(f"Se of {dist.name}({cnt})\t\t: {dist.se(cnt)}")
        print(f"Median of {dist.name}({cnt})\t\t: {dist.median(cnt)}\n")

# Постройка гистрограммы с шагом bins
fig, ax = plt.subplots(1, 1)
ax.hist(dists[0].rvs(100), bins=10)
ax.set_title("Uniform(100 elements)")
plt.show()