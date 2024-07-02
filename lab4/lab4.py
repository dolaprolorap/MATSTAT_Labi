import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm

from tabulate import tabulate
from matplotlib.colors import LinearSegmentedColormap

redwine_filename = "winequality-red.csv"
whitewine_filename = "winequality-white.csv"

wine_data = pd.read_csv(redwine_filename, sep=";")

def plot_wine_hist():
    wine_data.hist(figsize=(15, 25))
    plt.show()

def print_wine_tabulated_data():
    print(tabulate(wine_data.describe().round(4), headers='keys', tablefmt='pretty', stralign='center'), '\n')

def print_wine_tabulated_corr_matrix():
    print(tabulate(wine_data.corr()))

def plot_heatmap():
    cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
    sn.heatmap(wine_data.corr(), annot=True, cmap=cmap)
    plt.show()

def print_heatmap_max_min():
    corr = wine_data.corr()
    max_corr = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().idxmax()
    min_corr = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().idxmin()
    print(max_corr)
    print(min_corr)
    
def regression():
    corr = wine_data.corr()
    max_corr = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().idxmax()
    min_corr = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().idxmin()
    sn.regplot(data=wine_data, x=max_corr[0], y=max_corr[1])
    plt.show()
    sn.regplot(data=wine_data, x=min_corr[0], y=min_corr[1])
    plt.show()

def ols():
    x = wine_data.iloc[:, :-1]
    y = wine_data.iloc[:, -1]
    model = sm.OLS(y, x)
    res = model.fit()
    print(res.summary(), "\n")
    predict = res.predict(x.iloc[:3])
    print("Predict: ", predict)
    print("Reality: ", y.iloc[:3].values)

# plot_wine_hist()
# print_wine_tabulated_data()
# print_wine_tabulated_corr_matrix()
# plot_heatmap()
# print_heatmap_max_min()
# regression()
# ols()