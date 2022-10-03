import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the Glass Identification Data Set csv data using the Pandas library
filename = 'dataset/glass.cvs'
df = pd.read_csv(filename)

mean = []
std = []
median = []
range = []
corrcoef = []

for col in df:
    if col in ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]:
        mean.append(df[col].mean())
        std.append(df[col].std())
        median.append(df[col].median())
        range.append(df[col].max() - df[col].min())
        corrcoef.append(np.corrcoef(df[col], df['type'])[0][1])
        

df_stats = pd.DataFrame(np.array([mean, std, median, range, corrcoef]), columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], index=['mean', 'std', 'median', 'range', 'corrcoef'])
print(df_stats)

# Correlation coefficient
corr = df.corr(method='pearson')
corr.style.background_gradient(cmap='coolwarm')
plt.matshow(corr)
plt.show()